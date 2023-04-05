from flask import Flask, request, render_template
from blockchain import BlockChain
import json

# initial implementation - Zackiss on 3.19
app = Flask(__name__)


@app.route('/')
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
    else:
        response["status"] = 500
        response["info"] = "requesting with empty transactions"
    return json.dumps(response, ensure_ascii=False)


@app.route('/verify_img', methods=['POST', 'GET'])
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
    blockchain = BlockChain()
    app.run()
