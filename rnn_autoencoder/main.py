"""
    Inspiration 
    from
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html    
"""


from os import device_encoding
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rouge import Rouge

import config
from seq2seq import *
from language import *
from utils import *
from plot import *


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention=True, max_length=config.MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.DEVICE)

    loss = 0
    
    for input in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[input], encoder_hidden)
        encoder_outputs[input] = encoder_output[0, 0]

    decoder_input = torch.tensor([[config.SOS_TOKEN]], device=config.DEVICE)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < config.TFR else False

    if use_teacher_forcing:
        for input in range(target_length):
            if attention:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[input])
            decoder_input = target_tensor[input]
    else:
        for input in range(target_length):
            if attention:    
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[input])

            if decoder_input.item() == config.EOS_TOKEN:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iterations(input_language, output_language, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=1000, learning_rate=0.01, attention=True):
    import time
    start = time.time()

    plot_losses = []
    iterations = []
    i = 1
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensors_from_pair(input_language, output_language, random.choice(pairs)) for _ in range(n_iters)]
    
    criterion = nn.NLLLoss()

    for iteration in range(1, n_iters + 1):
        training_pair = training_pairs[iteration - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention=attention)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{time_since(start, iteration / n_iters)} ({iteration} {iteration / n_iters * 100}%) {print_loss_avg}')            
        
        if iteration % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            iterations.append(i)
            i += 1
            plot_loss_total = 0

    plot(plot_losses, iterations)


def evaluate(input_language, output_language, encoder, decoder, text, attention=True, max_length=config.MAX_LENGTH):
    with torch.no_grad():        
        input_tensor = tensor_from_text(input_language, text)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.DEVICE)

        for input in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[input], encoder_hidden)
            encoder_outputs[input] += encoder_output[0, 0]

        decoder_input = torch.tensor([[config.SOS_TOKEN]], device=config.DEVICE)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attns = torch.zeros(max_length, max_length)

        for input in range(max_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attns[input] = decoder_attn.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)            
            _, topi = decoder_output.data.topk(1)
            if topi.item() == config.EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_language.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluate_randomly(input_language, output_language, pairs, encoder, decoder, attention=True, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('<Text body>', pair[0], '<\Text body>\n')
        print('<Summary>', pair[1], '</Summary>\n')
        output_words = evaluate(input_language, output_language, encoder, decoder, pair[0], attention)
        output_sentence = ' '.join(output_words)
        print('<Output>', output_sentence, '</Output>\n')        


def rouge_scores(input_language, output_language, pairs, encoder, decoder, attention=True):    
    N = len(pairs)
    evaluator = Rouge() 
    
    rouge_vals = {
        'rouge-1': np.zeros((3, N)),
        'rouge-2': np.zeros((3, N)),
        'rouge-l': np.zeros((3, N))
    }

    rouge_1f_vs_len = [[], []]

    for i, pair in enumerate(pairs):
        output_words = evaluate(input_language, output_language, encoder, decoder, pair[0], attention=attention)
        output_sentence = ' '.join(output_words)
        if len(output_sentence) == 0 or len(pair[1]) == 0: continue
        rouge_score = evaluator.get_scores(pair[1], output_sentence)
        rouge_vals['rouge-1'][0][i] = rouge_score[0]['rouge-1']['f']
        rouge_vals['rouge-1'][1][i] = rouge_score[0]['rouge-1']['p']
        rouge_vals['rouge-1'][2][i] = rouge_score[0]['rouge-1']['r']
        rouge_vals['rouge-2'][0][i] = rouge_score[0]['rouge-2']['f']
        rouge_vals['rouge-2'][1][i] = rouge_score[0]['rouge-2']['p']
        rouge_vals['rouge-2'][2][i] = rouge_score[0]['rouge-2']['r']
        rouge_vals['rouge-l'][0][i] = rouge_score[0]['rouge-l']['f']
        rouge_vals['rouge-l'][1][i] = rouge_score[0]['rouge-l']['p']
        rouge_vals['rouge-l'][2][i] = rouge_score[0]['rouge-l']['r']
        rouge_1f_vs_len[0].append(len(output_words))
        rouge_1f_vs_len[1].append(rouge_vals['rouge-1'][0][i])

    rouge_avgs = {'rouge-1': np.zeros(3), 'rouge-2': np.zeros(3), 'rouge-l': np.zeros(3)}
    rouge_sums = {'rouge-1': np.zeros(3), 'rouge-2': np.zeros(3), 'rouge-l': np.zeros(3)}

    for key in rouge_vals.keys():
        for i in range(3):
            rouge_sums[key][i] = sum(rouge_vals[key][i])
            rouge_avgs[key][i] = (1 / N) * rouge_sums[key][i]    
            
    return rouge_avgs, rouge_sums, rouge_1f_vs_len


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'True':        
        print(f'Using Spotify data')
        training_data = pd.read_pickle(r'../final_data/spotify_train_422.pkl')
        X, Y = training_data['body'], training_data['episode_desc']
    else: 
        print(f'Using BC3 data')
        training_data = pd.read_pickle(r'../final_data/BC3_127.pkl')
        X, Y = training_data['body'], training_data['summary']                

    print(training_data.info())      

    # results = search_episode(training_data, 'Hello, my name is Kate Cocker. Are you the')

    # for r in results:
    #     print(r)

    config.MAX_LENGTH = len(max(list(training_data['body']), key=len))
    
    clean_body, clean_summary = clean_text(X), clean_text(Y)
    
    input_language, output_language, pairs = prepare_data(clean_body, clean_summary)
    idx = np.random.permutation(len(pairs))
    idx_train, idx_test = idx[:len(idx) - 15], idx[len(idx) - 15:]
    pairs_train, pairs_test = np.array(pairs)[idx_train], np.array(pairs)[idx_test]
    
    hidden_sizes = [128, 256, 512]
    hidden_size = 256
    attention = False
    n_iterations = 7000
    lr = 0.01
    
    for hidden_size in hidden_sizes:
        if attention:
            print('Using attention mechanism')
            encoder = EncoderRNN(input_language.n_words, hidden_size).to(config.DEVICE)        
            attn_decoder = AttnDecoderRNN(hidden_size, output_language.n_words, dropout_p=0.1).to(config.DEVICE)        
            train_iterations(input_language, output_language, pairs_train, encoder, attn_decoder, n_iterations, print_every=100, plot_every=1000, learning_rate=lr)
                    
            evaluate_randomly(input_language, output_language, pairs_test, encoder, attn_decoder)                
            rouge_test, _, len_vs_score = rouge_scores(input_language, output_language, pairs_test, encoder, attn_decoder)
            print(f'Rouge scores for test data\n{rouge_test}')        
        else:
            print('Not using attention mechanism')
            encoder = EncoderRNN(input_language.n_words, hidden_size).to(config.DEVICE)
            decoder = DecoderRNN(hidden_size, output_language.n_words).to(config.DEVICE)
            train_iterations(input_language, output_language, pairs_train, encoder, decoder, n_iterations, print_every=100, plot_every=1000, learning_rate=lr, attention=False)
                            
            evaluate_randomly(input_language, output_language, pairs_test, encoder, decoder, attention=False)                
            rouge_test, _, len_vs_score = rouge_scores(input_language, output_language, pairs_test, encoder, decoder, attention=False)
            print(f'Rouge scores for test data\n{rouge_test}')        

    scatter_plot(len_vs_score)


if __name__ == '__main__':
    main()