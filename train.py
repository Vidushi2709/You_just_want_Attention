import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from model import build_a_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm #shows the live progress bar
import warnings
import torch
import torch.nn as nn
import torchmetrics 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    #precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    #initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True: # loop to generate op tasks
        if decoder_input.size(1) == max_len:
            break

        #casual mask for decoder
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #decode next token
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #project o/p to vocab
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        #append predicted tokens
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step,writer, num_examples=2):
    model.eval()
    count=0 #keep track of no of batches processed 
    src_txt=[]
    expects=[]
    preds=[]

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80   
    
    with torch.no_grad(): #no gradients calcluation during val
        for batch in validation_ds:
            count +=1
            encoder_ip= batch['encoder_ip'].to(device)
            encoder_mask= batch['encoder_mask'].to(device)
            assert encoder_ip.size(0)==1 #batch size ==1 for val always
            model_out= greedy_decode(model, encoder_ip, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_txts= batch['src_txt'][0]
            tgt_txts= batch['tgt_txt'][0]
            model_out_txt= tokenizer_tgt.decode(model_out.detach().cpu().numpy()) #model op decoded back into txt using .decode

            #store and print results
            src_txt.append(src_txts)
            expects.append(tgt_txts)
            preds.append(model_out_txt)

            print_msg('-' * console_width)
            print_msg(f"{'SOURCE: ':>12}{src_txt}")
            print_msg(f"{'TARGET: ':>12}{tgt_txts}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_txt}")

            if count==num_examples:
                print_msg('-'*console_width)
                break

        if writer:
                # Evaluate the character error rate
                # Compute the char error rate 
                metric = torchmetrics.CharErrorRate()
                cer = metric(preds, expects)
                writer.add_scalar('validation cer', cer, global_step)
                writer.flush()

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(preds, expects)
                writer.add_scalar('validation wer', wer, global_step)
                writer.flush()

                # Compute the BLEU metric
                metric = torchmetrics.BLEUScore()
                bleu = metric(preds, expects)
                writer.add_scalar('validation BLEU', bleu, global_step)
                writer.flush()


config = {
    "datasource": "wmt14",
    "lang_src": "en",
    "lang_tgt": "de"
}
def get_ds(config):
    ds_raw= load_dataset( config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    #build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    #find max len of each sentence in src and tgt so that later model i/ps can be padded accordingly
    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids= tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids= tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src= max(max_len_src, len(src_ids))
        max_len_tgt= max(max_len_tgt, len(tgt_ids))

    #dataloader for training and validation
    train_dataloader= DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader= DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model= build_a_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], d_model=config['d_model'])
    return model
#config["seq_len"]: Maximum seque  nce length for both source and target sentences.
#d_model: Dimensionality of token embeddings and internal Transformer layers.

def train_the_damn_model(config):
    device = torch.device("cpu")
    print("Using device: CPU")

    #weights folders exists right?
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True,exist_ok=True)

    #load data, model, optimizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt= get_ds(config)
    model= get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer= SummaryWriter(config['experiment_name']) #for tensorboard loggs
    optimizer= torch.optim.Adam(model.parameters(), lr= config['lr'], eps= 1e-9)

    #the block checks iif user wants to resume training from prev saved checkpoint
    #otherwise start training from scratch
    initial_epoch=0
    global_step=0
    preload= config['preload']
    model_filename= latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    #if file == found restor stuff
    if model_filename:
        state= torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch= state['epoch']+1
        global_step= state['global_step']

    #loss function ignores padding tokens and uses label smoothing to regularize training and reduce overconfidence (hehe)
    loss_fn= nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1,
    ).to(device)

    #train train train choo choo choo
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator= tqdm(train_dataloader, desc=f"Processing Epoch choo choo{epoch:02d}")

    #batch wise forward +backward pass
    #send ip to model
    #runs ecnoder->decoder->final projection layer
    #comapres preds with ground truth using loss loss
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        
            #logging-> backpropogation->weight update
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step+=1

        #validation at end of epochsss
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        #save the mdoel at end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_the_damn_model(config)

