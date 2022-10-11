from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer, models, trainers
from pathlib import Path


def bert_walk_tokenizer(data):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    my_tokens = Path("artifacts/tokenizer.pt")
    if my_tokens.exists():
        print("Reading Tokens from Disk")
        tokenizer = tokenizer.from_file(my_tokens._str)
    else:
        print("training tokenizer!")
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[MASK]", "[CLS]", "[PAD]"])
        tokenizer.enable_padding()
        tokenizer.train_from_iterator(map(lambda x: x.split(), data), trainer=trainer)
        tokenizer.save(my_tokens._str)
    return tokenizer