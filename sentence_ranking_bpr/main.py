import pickle
from sklearn.model_selection import train_test_split

from ranking_models import *

if __name__ == '__main__':
    args={
        'modelname1': 'bert-base-uncased', 
        'modelname2': 'allenai/scibert_scivocab_uncased',
        'debug_mode': True,
        'hidden_size': 256,
        'dropout': 0.2,
        'batch_size': 10,
        'negative_sampling_size': 10,
        'debug_batch_size': 1,
        'epochs': 2,
        'lr': 0.001,
        'seed': 1,
        'early_stop_times': 5,
        'sentence_ranking_file': '../results/task1/ranking/round_2_ranking_results_test_set.csv',  # file name to save sentence_ranking 
        'device': 0
    }

    with open('../results/task1/pseudo_label.pkl', 'rb') as f:
        sentences = pickle.load(f)
    
    labeled = sentences.loc[sentences['label']==1]
    test = sentences.loc[sentences['label']==-1]
    train, val = train_test_split(labeled, test_size=0.2)

    if args['debug_mode']:
        test = test[:20]

    train_sent, val_sent, test_sent = map(lambda x: x.abstract.values, [train, val, test])
    train_prob, val_prob, test_prob = map(lambda x: x.prob.values, [train, val, test])

    # pass the probability
    train_labels, val_labels, test_labels = train_prob, val_prob, test_prob

    num_labels = 1

    device = torch.device('cuda:'+str(args['device']))
    batch_size = args['debug_batch_size'] if args['debug_mode'] else args['batch_size']
    num_negative = 1 if args['debug_mode'] else args['negative_sampling_size']

    base_bert_name = args['modelname1']
    sci_bert_name = args['modelname2']

    if args['debug_mode']:
        train_sent, _, train_labels, _ = train_test_split(train_sent, train_labels, train_size=100)
        val_sent, _, val_labels, _ = train_test_split(val_sent, val_labels, train_size=100)

    processed = bertrnn_process(train_sent, 
                                val_sent, 
                                test_sent, 
                                train_labels, 
                                val_labels, 
                                test_labels,
                                batch_size, 
                                num_negative,
                                base_bert_name=base_bert_name,
                                sci_bert_name=sci_bert_name)
    nor_loader, sci_loader, nor_train_collator, nor_val_collator, sci_train_collator, sci_val_collator = processed

    model= BERTRNN(num_labels,
                   hidden_size=args['hidden_size'],
                   dropout=args['dropout'],
                   base_bert_name=args['modelname1'],
                   sci_bert_name=args['modelname2']).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criteria = BPRLoss()

    best_val_loss = None
    early_stoping_counter = 0
    for epoch in range(1, args['epochs'] + 1):
        epoch_loss = train_(model, criteria, optimizer, nor_loader[0], sci_loader[0], nor_train_collator, sci_train_collator, epoch, device)

        val_loss = validate_(model, criteria, optimizer, nor_loader[1], sci_loader[1], nor_val_collator, sci_val_collator, epoch, device)

        if epoch == 1 or val_loss < best_val_loss:
            print('- new best loss: {}'.format(val_loss))
            best_val_loss = val_loss
            early_stoping_counter = 0
            test_and_save_results(model, device, nor_loader[2], sci_loader[2], test, args['sentence_ranking_file'])
        else:
            early_stoping_counter += 1

        if early_stoping_counter >= args['early_stop_times']:
            break

    print ('- early stopping {} epochs without improvement'.format(epoch))