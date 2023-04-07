from database import save_block_to_chain, get_all_blocks_from_chain
import time
import hashlib
import json
import random


# initial implementation - Zackiss on 3.19
class BlockChain(object):
    def __init__(self):
        self.cur_capacity = 0
        self.max_capacity = 10
        self.chain = self.get_chain()
        self.cur_transactions = []
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
        self.cur_capacity += 1

    def add_transaction(self, transaction):
        assert self.cur_capacity <= self.max_capacity
        # if block out of capacity
        if self.cur_capacity == self.max_capacity:
            prev_block = self.get_chain_fin()
            proof = self.proof_work(prev_block["proof"])
            self.new_block(proof, previous_hash=self.hash(prev_block))
        # if block not full, add transaction to the block
        else:
            self.cur_transactions.append(transaction)

    def verify_transaction(self, pic) -> bool:
        self.get_chain()
        for block in self.chain:
            for transaction in block["transactions"]:
                if pic == transaction["pic_hash"]:
                    return True
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
