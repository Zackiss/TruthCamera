from flask import Flask, request, render_template
from blockchain import BlockChain 
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

def check_transcation_format(transcation: dict):
  # verify the transcation in format: {depth_data, pic_hash}
  if not isinstance(transcation, dict):
      return False
  for dam in ["depth_data", "pic_hash"]:
    if dam not in list(transcation.keys()):
      return False
  return True

@app.route('/add_trans',methods = ['POST', 'GET'])
def add_block():
  transcation = None
  response = {}
  
  if len(request.args):
    response["status"] = 200
    response["info"] = "transcation received"
    transcation = request.args.to_dict().get("transcation", None)
    
    # if block not in correct format
    if transcation is None or not check_transcation_format(transcation):
      response["status"] = 300
      response["info"] = "transcation received with incorrect format"
    # if everything works well, we shall add transcation to blockchain
    else:
      blockchain.add_transcation(transcation)
  else:
    response["status"] = 500
    response["info"] = "requesting with empty transcations"
  return json.dump(response, ensure_ascii=False)

@app.route('/verify_img',methods = ['POST', 'GET'])
def verify_block():
  response = {}
  if len(request.args):
    response["status"] = 200
    response["info"] = "pic received"
    response["result"] = None
    pic = request.args.to_dict().get("pic_hash", None)
    if pic is None:
      response["status"] = 300
      response["info"] = "transcation received with incorrect format"
    # if everything works well, we shall verify transcation
    else:
      response["result"] = blockchain.verify_transcation(pic)
  else:
    response["status"] = 500
    response["info"] = "requesting with empty transcations"
  return json.dump(response, ensure_ascii=False)
  
if __name__ == '__main__':
    blockchain = BlockChain()
    app.run()