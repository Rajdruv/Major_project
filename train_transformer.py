import pandas as pd
import Levenshtein
import tokenizers
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, LineByLineTextDataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling


def load_data():
    data = pd.read_csv("all_data.csv")
    return data


def convert_second(x):
    temp = (
        str(x.year)
        + "-"
        + str(x.month)
        + "-"
        + str(x.day)
        + " "
        + str(x.hour)
        + "-"
        + str(x.minute)
        + "-"
        + str(x.second)
    )
    return temp


def find_closest_string(target):
    string_dict = {
        """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part, supplier, partsupp, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 15 and p_type like &apos;%BRASS&apos; and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = &apos;EUROPE&apos; and ps_supplycost = (select min(ps_supplycost) from partsupp, supplier, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = &apos;EUROPE&apos; ) order by s_acctbal desc, n_name, s_name, p_partkey;""": "a",
        """select ps_partkey, sum(ps_supplycost * ps_availqty) as value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = &apos;RUSSIA&apos; group by ps_partkey having sum(ps_supplycost * ps_availqty) &gt; (select sum(ps_supplycost * ps_availqty) * 0.0001 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = &apos;RUSSIA&apos;) order by value desc;""": "b",
        """select * from part p where p.p_size &gt; 20;""": "c",
        """select DISTINCT (p_container) from part p ;""": "d",
        """select * from supplier s where s_nationkey &gt; 20;""": "e",
        """select * from supplier s where s_nationkey &gt; 20;""": "f",
        """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part, supplier, partsupp, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 23 and p_type like &apos;%STEEL&apos; and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = &apos;EUROPE&apos; and ps_supplycost = (select min(ps_supplycost) from partsupp, supplier, nation, region where dp_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = &apos;EUROPE&apos;) order by s_acctbal desc, n_name, s_name, p_partkey;""": "g",
        """select sum(l_extendedprice*l_discount) as revenue from lineitem where l_shipdate &gt;= date &apos;1993-02-03&apos; and l_shipdate &lt; date &apos;1993-02-04&apos; + interval &apos;1&apos; year and l_discount between 0.03 - 0.01 and 0.05 + 0.01 and l_quantity &lt; 24;""": "h",
        """select ps_partkey, sum(ps_supplycost * ps_availqty) as value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = &apos;FRANCE&apos; group by ps_partkey having sum(ps_supplycost * ps_availqty) &gt; (select sum(ps_supplycost * ps_availqty) * 0.00003 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = &apos;FRANCE&apos;) order by value desc;""": "i",
        """select c_count, count(*) as custdist from ( select c_custkey, count(o_orderkey) from customer left outer join orders on c_custkey = o_custkey and o_comment not like &apos;%special%packages%&apos; group by c_custkey) as c_orders (c_custkey, c_count) group by c_count order by custdist desc, c_count desc;""": "j",
        """select o_orderpriority, count(*) as order_count from orders where o_orderdate &gt;= date &apos;1993-02-01&apos; and o_orderdate &lt; date &apos;1993-04-03&apos; + interval &apos;3&apos; month and exists ( select * from lineitem where l_orderkey = o_orderkey and l_commitdate &lt; l_receiptdate) group by o_orderpriority order by o_orderpriority;""": "k",
        """select * from orders where o_orderstatus = &apos;O&apos;""": "l",
    }
    string_list = list(string_dict.keys())
    closest_string = None
    min_distance = float("inf")

    for string in string_list:
        distance = Levenshtein.distance(target, string)
        if distance < min_distance:
            min_distance = distance
            closest_string = string

    return string_dict[closest_string]


def prepare_data():
    data = load_data()
    data.dropna(subset=['query'], inplace=True)
    # breakpoint()
    data["represent"] = data["query"].apply(find_closest_string)
    data["time"] = pd.to_datetime(data["time"])
    data["tosecond"] = data["time"].apply(convert_second)
    result = data.groupby("tosecond")["represent"].agg("".join).reset_index()
    all_string = result.represent.tolist()
    with open("all_words.txt", "w") as file:
        for string in all_string:
            file.write(string + "\n")


def train_tokenize():
    bwpt = tokenizers.BertWordPieceTokenizer(vocab=None)
    bwpt.train(
        files=["all_words.txt"], vocab_size=5000, min_frequency=3, limit_alphabet=1000
    )

    bwpt.save("customtokenizer.json")


def train_transformer_model():
    tokenizer = BertTokenizer.from_pretrained("customtokenizer.json")
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="all_words.txt",
        block_size=128,  # maximum sequence length
    )
    config = BertConfig(
        vocab_size=5000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=512,
    )

    model = BertForMaskedLM(config)
    print("No of parameters: ", model.num_parameters())

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="model_output",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("major_project_saved_model")


def train_transformer_main():
    prepare_data()
    train_tokenize()
    train_transformer_model()
