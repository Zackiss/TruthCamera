from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin
import time
import hashlib
import random
import json
import pickle

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chain.sqlite"
db = SQLAlchemy(app)
cors = CORS(app)


# update implementation - Zackiss on 4.7
# ---------------BLOCKCHAIN part--------------------
class BlockChain(object):
    def __init__(self):
        self.cur_capacity = 0
        self.max_capacity = 1
        self.cur_transactions = []
        self.chain = [{
            "index": 0,
            "timestamp": time.time(),
            "transactions": self.cur_transactions,
            "proof": 12100,
            "previous_hash": 12100
        }]
        self.first_hash = self.hash({
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "transactions": self.cur_transactions,
            "proof": 12100,
            "previous_hash": 12100
        })

    def get_chain_fin(self):
        return self.chain[-1]

    def get_chain(self) -> list[dict]:
        self.chain = get_all_blocks_from_chain()
        return self.chain

    def new_block(self, proof, previous_hash=None):
        # create a new block and save it to the chain
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "transactions": self.cur_transactions,
            "proof": proof,
            "previous_hash": previous_hash if previous_hash else self.first_hash
        }
        save_block_to_chain(block)
        self.chain = self.get_chain()
        self.cur_capacity = 0
        print("adding finished")

    def add_transaction(self, transaction):
        assert self.cur_capacity <= self.max_capacity
        # if block out of capacity
        if self.cur_capacity == self.max_capacity:
            print("adding to blockchain")
            prev_block = self.get_chain_fin()
            proof = self.proof_work(prev_block["proof"])
            self.new_block(proof, previous_hash=self.hash(prev_block))
            
        # if block not full, add transaction to the block
        else:
            self.cur_transactions.append(transaction)
            self.cur_capacity += 1
            print("buffering: " + str(self.cur_transactions))
            print("current capacity reach to: " + str(self.cur_capacity))
            print("max capacity: " + str(self.max_capacity))

    def verify_transaction(self, pic) -> bool:
        self.get_chain()
        print("obtaining: " + str(pic))
        print("verifying with chain: " + str(self.chain))
        for block in self.chain:
            for transaction in block["transactions"]:
                if pic == '"' + transaction["pic_hash"] + '"':
                    print("hash found!")
                    return True
        print("hash not found!")
        return False

    def hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_work(self, prev_proof):
        proof = random.randint(0, 150)
        while not self.verify_proof(prev_proof, proof):
            proof += random.randint(0, 150)
        return proof

    @staticmethod
    def verify_proof(last_proof, proof) -> bool:
        guess = f"{last_proof}{proof}".encode()
        hashy = hashlib.sha256(guess).hexdigest()
        return hashy[:3] == "042"
   

blockchain = BlockChain()
    
# --------------TIPS---------------------
# you shall post or get your request at addresses: ip/add_trans, ip/verify_img
# you shall upload json with format {"transaction": {"depth_data": depth data, "pic_hash": pic hash}}
# you shall set up the front page at templates/index.html or redirect to other services


# update implementation - Zackiss on 4.7
# --------------DATABASE part---------------------
class Chain(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attributes = db.Column(db.PickleType)


def save_block_to_chain(block: dict):
    with app.app_context():
        db.create_all()
        db.session.add(Chain(attributes=block))
        db.session.commit()


def get_all_blocks_from_chain() -> list:
    blocks = []
    with app.app_context():
        db.create_all()
        for block in db.session.query(Chain.attributes):
            print(block[0])
            blocks.append(block[0])
    return blocks


# update implementation - Zackiss on 4.7
# ---------------FLASK part--------------------
@app.route('/')
@cross_origin()
def index():
    return render_template("index.html")


def check_transaction_format(transaction: dict):
    # verify the transaction in format: {depth_data, pic_hash}
    if not isinstance(transaction, dict):
        return False
    for dam in ["depth_data", "pic_hash"]:
        if dam not in list(transaction.keys()):
            return False
    return True


@app.route('/add_trans', methods=['POST', 'GET'])
@cross_origin()
def add_block():
    transaction = None
    response = {}

    if len(request.args):
        response["status"] = 200
        response["info"] = "transaction received"
        trans_str = request.args.to_dict().get("transaction", None)
        transaction = json.loads(trans_str)

        # if block not in correct format
        if transaction is None or not check_transaction_format(transaction):
            response["status"] = 300
            response["info"] = "transaction received with incorrect format"
        # if everything works well, we shall add transaction to blockchain
        else:
            blockchain.add_transaction(transaction)
            response["info"] = "transaction received and saved to buffer"
    else:
        response["status"] = 500
        response["info"] = "requesting with empty transactions"
    return json.dumps(response, ensure_ascii=False)


@app.route('/verify_img', methods=['POST', 'GET'])
@cross_origin()
def verify_block():
    response = {}
    if len(request.args):
        response["status"] = 200
        response["info"] = "pic received"
        response["result"] = False
        pic = request.args.to_dict().get("pic_hash", None)
        if pic is None:
            response["status"] = 300
            response["info"] = "transaction received with incorrect format"
        # if everything works well, we shall verify transaction
        else:
            response["result"] = blockchain.verify_transaction(pic)
    else:
        response["status"] = 500
        response["info"] = "requesting with empty transaction"
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    app.run()


