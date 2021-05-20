"""
    Inspiration 
    from
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html    
"""


from os import device_encoding
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


"""
    TODO:
    Now
        [x] Add ROUGE metrics
        [x] Plot loss
        [x] Extend ROUGE metrics to rouge-1, rouge-2 and rouge-l
        
    Later
        [] Replace the embeddings with pre-trained word embeddings such as word2vec or GloVe
        [] Try with more layers, more hidden units, and more sentences. Compare the training time and results.
"""


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=config.MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.DEVICE)

    loss = 0

    # print(f'input length: {input_length}')
    for input in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[input], encoder_hidden)
        encoder_outputs[input] = encoder_output[0, 0]

    decoder_input = torch.tensor([[config.SOS_TOKEN]], device=config.DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < config.TFR else False

    if use_teacher_forcing:
        for input in range(target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[input])
            decoder_input = target_tensor[input]
    else:
        for input in range(target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[input])

            if decoder_input.item() == config.EOS_TOKEN:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iterations(input_language, output_language, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
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
    
    # negative log-likelihood loss
    # alternatively we can remove softmax layer and use cross-entropy loss
    criterion = nn.NLLLoss()

    for iteration in range(1, n_iters + 1):
        training_pair = training_pairs[iteration - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{time_since(start, iteration / n_iters)} ({iteration} {iteration / n_iters * 100}%) {print_loss_avg}')
            # print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
            #                              iter, iter / n_iters * 100, print_loss_avg))
        
        if iteration % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            iterations.append(i)
            i += 1
            plot_loss_total = 0

    plot(plot_losses, iterations)


def evaluate(input_language, output_language, encoder, decoder, text, max_length=config.MAX_LENGTH):
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
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attns[input] = decoder_attn.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == config.EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_language.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attns[:input + 1]


def evaluate_randomly(input_language, output_language, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_language, output_language, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def rouge_scores(input_language, output_language, pairs, encoder, decoder):    
    N = len(pairs)
    evaluator = Rouge() 
    rouge_avgs = {
        'rouge-1': np.zeros(3),
        'rouge-2': np.zeros(3),
        'rouge-l': np.zeros(3)
    }
    
    rouge_sums = {
        'rouge-1': np.zeros((3, N)),
        'rouge-2': np.zeros((3, N)),
        'rouge-l': np.zeros((3, N))
    }
    

    for i, pair in enumerate(pairs):
        output_words, attns = evaluate(input_language, output_language, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        rouge_score = evaluator.get_scores(pair[1], output_sentence)
        rouge_avgs['rouge-1'][0] += (1 / N) * rouge_score[0]['rouge-1']['f']
        rouge_avgs['rouge-1'][1] += (1 / N) * rouge_score[0]['rouge-1']['p']
        rouge_avgs['rouge-1'][2] += (1 / N) * rouge_score[0]['rouge-1']['r']
        rouge_avgs['rouge-2'][0] += (1 / N) * rouge_score[0]['rouge-2']['f']
        rouge_avgs['rouge-2'][1] += (1 / N) * rouge_score[0]['rouge-2']['p']
        rouge_avgs['rouge-2'][2] += (1 / N) * rouge_score[0]['rouge-2']['r']
        rouge_avgs['rouge-l'][0] += (1 / N) * rouge_score[0]['rouge-l']['f']
        rouge_avgs['rouge-l'][1] += (1 / N) * rouge_score[0]['rouge-l']['p']
        rouge_avgs['rouge-l'][2] += (1 / N) * rouge_score[0]['rouge-l']['r']

        rouge_sums['rouge-1'][0][i] = rouge_score[0]['rouge-1']['f']
        rouge_sums['rouge-1'][1][i] = rouge_score[0]['rouge-1']['p']
        rouge_sums['rouge-1'][2][i] = rouge_score[0]['rouge-1']['r']
        rouge_sums['rouge-2'][0][i] = rouge_score[0]['rouge-2']['f']
        rouge_sums['rouge-2'][1][i] = rouge_score[0]['rouge-2']['p']
        rouge_sums['rouge-2'][2][i] = rouge_score[0]['rouge-2']['r']
        rouge_sums['rouge-l'][0][i] = rouge_score[0]['rouge-l']['f']
        rouge_sums['rouge-l'][1][i] = rouge_score[0]['rouge-l']['p']
        rouge_sums['rouge-l'][2][i] = rouge_score[0]['rouge-l']['r']
        
    print(f'rouge-1: {rouge_avgs}')
    return rouge_avgs, rouge_sums
    

# def main():
#     training_data = pd.read_pickle(r'../data/dataframes/spotify_train_vectors.pkl')
#     test_data = pd.read_pickle(r'../data/dataframes/spotify_test_vectors.pkl')
#     print(training_data.info())    
#     print(test_data.info())    
    
#     X_train, Y_train = training_data['body'], training_data['episode_desc']
#     X_test, Y_test = test_data['body'], test_data['episode_desc']
    
#     clean_body_train, clean_summary_train = clean_text(X_train), clean_text(Y_train)
#     clean_body_test, clean_body_summary = clean_text(X_test), clean_text(Y_test)
    
#     # print('Transcription')
#     # print(clean_body[0])
#     # print('Summary')
#     # print(clean_summary[0])
    
#     input_language, output_language, pairs_train = prepare_data(clean_body_train, clean_summary_train)
    
#     pairs_test = [[clean_body_test[i], clean_body_summary[i]] for i in range(len(clean_body_test))]
    
#     for pair in pairs_test:
#         input_language.add_sentence(pair[0])
#         output_language.add_sentence(pair[1])

#     # input_tensor, target_tensor = tensors_from_pair(input_language, output_language, pairs[0])
    
#     hidden_size = 256
#     encoder = EncoderRNN(input_language.n_words, hidden_size).to(config.DEVICE)
#     attn_decoder = AttnDecoderRNN(hidden_size, output_language.n_words, dropout_p=0.1).to(config.DEVICE)

#     n_iterations = 500
#     train_iterations(input_language, output_language, pairs_train, encoder, attn_decoder, n_iterations, print_every=100)
#     evaluate_randomly(input_language, output_language, pairs_test, encoder, attn_decoder)


def main():
    data = pd.read_pickle(r'../data/dataframes/wrangled_BC3_df.pkl')
    print(data.info())
    X = data['body']
    Y = data['summary']

    clean_body = clean_text(X)
    clean_summary = clean_text(Y)

    input_language, output_language, pairs = prepare_data(clean_body, clean_summary)
    train_pairs = pairs[0:209]
    test_pairs = pairs[209:259]

    
    # hidden_size = 128
    hidden_size = 256
    # hidden_size = 512
    # hidden_sizes = [128, 256, 512]

    # for hidden_size in hidden_sizes:
    encoder = EncoderRNN(input_language.n_words, hidden_size).to(config.DEVICE)
    attn_decoder = AttnDecoderRNN(hidden_size, output_language.n_words, dropout_p=0.1).to(config.DEVICE)

    n_iterations = 25000
    learning_rate = 0.01
    train_iterations(input_language, output_language, train_pairs, encoder, attn_decoder, n_iterations, print_every=100, plot_every=100, learning_rate=learning_rate)
    evaluate_randomly(input_language, output_language, test_pairs, encoder, attn_decoder)
    rouge_scores(input_language, output_language, test_pairs, encoder, attn_decoder)


if __name__ == '__main__':
    main()

"""
    BC3 main

    def main():
        data = pd.read_pickle(r'../data/dataframes/wrangled_BC3_df.pkl')
        print(data.info())
        X = data['body']
        Y = data['summary']

        clean_body = clean_text(X)
        clean_summary = clean_text(Y)
        input_language, output_language, pairs = prepare_data(clean_body, clean_summary)

        # input_tensor, target_tensor = tensors_from_pair(input_language, output_language, pairs[0])
        
        hidden_size = 256
        encoder = EncoderRNN(input_language.n_words, hidden_size).to(config.DEVICE)
        attn_decoder = AttnDecoderRNN(hidden_size, output_language.n_words, dropout_p=0.1).to(config.DEVICE)

        n_iterations = 75000
        train_iterations(input_language, output_language, pairs, encoder, attn_decoder, n_iterations, print_every=100)
        evaluate_randomly(input_language, output_language, pairs, encoder, attn_decoder)

"""