pragma solidity ^0.8.0;

contract MyContract {
    string[] public myStrings;
    
    function addString(string memory newString) public {
        myStrings.push(newString);
    }
    
    function checkString(string memory query) public view returns (bool) {
        for (uint i = 0; i < myStrings.length; i++) {
            if (keccak256(abi.encodePacked(myStrings[i])) == keccak256(abi.encodePacked(query))) {
                return true;
            }
        }
        return false;
    }
}