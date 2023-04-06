# Replace the database operation of sqlite storage area information. 
# Use web3 to interact with Ethereum instead of directly manipulating the database.

from web3 import Web3
import sqlite3

def push_to_Ethereum(pic_hash):
    # Connect to the Ethereum network
    web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/<YOUR_INFURA_PROJECT_ID>')) # There will be a infura project id in the real usage.

    # Create new transaction
    transaction = {
        'from': '0xAb8483F64d9C6d1EcF9b849Ae677dD3315835cb2', # Temporary address of the sender
        'to': '0xAb8483F64d9C6d1EcF9b849Ae677dD3315835cb2', # Temporary address of the receiver
        'value': web3.toWei('0', 'ether'),  
        'gas': 21000,     
        'gasPrice': web3.toWei('50', 'gwei'),  
        'nonce': web3.eth.getTransactionCount('0xAb8483F64d9C6d1EcF9b849Ae677dD3315835cb2'),  
    }

    # Use the private key to sign the transaction
    signed_transaction = web3.eth.account.sign_transaction(transaction, private_key='0x...')

    contract_address = '0xD7ACd2a9FD159E69Bb102A1ca21C9a3e3A5F771B' # The contract address of the smart contract contract.sol in the same directory
    contract_abi = [{'constant': False, 'inputs': [{'name': 'data', 'type': 'string'}], 'name': 'addData', 'outputs': [], 'payable': False, 'stateMutability': 'nonpayable', 'type': 'function'}]

    # Get the contract instance
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    # Send the transaction
    tx_hash = contract.functions.addString(pic_hash).transact({'from': '0xAb8483F64d9C6d1EcF9b849Ae677dD3315835cb2', 'gas': 100000, 'gasPrice': web3.toWei('50', 'gwei')})

    # Wait for the transaction to be mined
    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

    # Check whether the transaction is successful
    if tx_receipt.status == 1:
        print('The image has already been added to the Ethereum network.')
    else:
        print('The image has not been added to the Ethereum network, please try again.')

    # Store the record of the transaction into our team's database
    conn = sqlite3.connect(' tx_receipt.db ')
    c = conn.cursor()
    c.execute("""CREATE TABLE receipts
    (tx_hash text, block_number integer)""")
    c.execute("INSERT INTO receipts VALUES ('{}','{}')".format(tx_receipt['transactionHash'], tx_receipt['blockNumber']))

    conn.commit()
    conn.close()