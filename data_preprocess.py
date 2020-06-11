import os
import torch
from tqdm import tqdm
import urllib.request
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import squad_convert_examples_to_features, AlbertTokenizer



data_dir = './data/'
model_name = 'albert-base-v2'
max_seq_length = 384
doc_stride = 128
max_query_length = 15
threads = 1
version_2_with_negative = True


tokenizer = AlbertTokenizer.from_pretrained(model_name)
input_dir = data_dir

print('downloading dataset')
def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])
def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)
        
for d in ('train', 'dev', 'test'):
    url = 'https://github.com/chrischute/squad/data/{}-v2.0.json'.format(d)
    output_path = url_to_data_path(url)
    if not os.path.exists(output_path):
            print(f'Downloading {d}...')
            download_url(url, output_path)

print("Creating features from dataset file at {}".format(input_dir))

processor = SquadV2Processor() if version_2_with_negative else SquadV1Processor()
for d in ('train', 'dev', 'test'):
    evaluate = d != 'train'
    cached_features_file = '.data/cached_{}'.format(d)
    examples = processor.get_dev_examples(data_dir, filename='{}-v2.0.json'.format(d))
    
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=threads,
    )
    
    print("Saving features into cached file {}".format(cached_features_file))
    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
