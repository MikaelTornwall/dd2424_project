from os import device_encoding
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config
from seq2seq import *
from language import *
from utils import *


"""
    TODO:
        - Replace the embeddings with pre-trained word embeddings such as word2vec or GloVe
        - Try with more layers, more hidden units, and more sentences. Compare the training time and results.
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

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{time_since(start, iteration / n_iters)} ({iteration} {iteration / n_iters * 100}%) {print_loss_avg}')
            # print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
            #                              iter, iter / n_iters * 100, print_loss_avg))
        # if iteration % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

    # show_plot(plot_losses)

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

    train_iterations(input_language, output_language, pairs, encoder, attn_decoder, 5000, print_every=100)
    evaluate_randomly(input_language, output_language, pairs, encoder, attn_decoder)


if __name__ == '__main__':
    main()