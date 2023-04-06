from web3 import Web3

def pull_from_Ethereum(target_hash):
    web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/<YOUR_INFURA_PROJECT_ID>')) # There will be a infura project id in the real usage.

    contract_address = '0xD7ACd2a9FD159E69Bb102A1ca21C9a3e3A5F771B'
    contract_abi = [{'constant': True, 'inputs': [{'name': 'query', 'type': 'string'}], 'name': 'checkString', 'outputs': [{'name': '', 'type': 'bool'}], 'payable': False, 'stateMutability': 'view', 'type': 'function'}]

    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    target_hash = '0x2f205bf1e3492fa35ab86279d7e0f12e1fcc8251' # The hash of image you want to check
    result = contract.functions.checkString(target_hash).call()

    print(result) # Prints True or False