from database import save_block_to_chain, get_all_blocks_from_chain
from time import time
import hashlib
import json
import random

# initial implementation - Zackiss on 3.19
class Blockchain(object):
  def __init__(self):
    self.cur_capacity = 0
    self.max_capacity = 10
    self.chain = self.get_chain()
    self.cur_transations = []
    self.first_hash = self.hash({
      "index": len(self.chain) + 1,
      "timestamp": time.time(),
      "transcations": self.cur_transations,
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
      "transcations": self.cur_transations,
      "proof": proof,
      "previous_hash": previous_hash if previous_hash else self.first_hash
    }
    save_block_to_chain(block)
    self.chain = self.get_chain()
    self.cur_capacity += 1

  def add_trancation(self, transation):
    assert self.cur_capacity <= self.max_capacity
    # if block out of capacity
    if self.cur_capacity == self.max_capacity:
      prev_block = self.get_chain_fin()
      proof = self.proof_work(prev_block["proof"])
      self.new_block(proof, previous_hash=self.hash(prev_block))
    # if block not full, add transation to the block
    else:
      self.cur_transations.append(transation)

  def verify_transcation(self, pic) -> bool:
    self.get_chain()
    for block in self.chain:
      if pic in block["transcation"]["pic_hash"]:
        return True
    return False

  def hash(self, block):
    block_string = json.dump(block, sort_keys=True).encode()
    return hashlib.sha256(block_string).nextdigest()

  def proof_work(self, prev_proof):
    proof = random.randint(0, 150)
    while not self.verif_proof(prev_proof, proof):
      proof += random.randint(0, 150)
    return proof

  def verify_proof(self, last_proof, proof) -> bool:
    guess = f"{last_proof}{proof}".encode()
    hash = hashlib.sha256(guess).hexdigest()
    return hash[:3] == "042"
