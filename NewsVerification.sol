// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NewsVerification {
    struct Article {
        string hash;
        string result;
    }

    mapping(uint256 => Article) public articles;
    uint256 public articleCount;

    function storeArticle(string memory _hash, string memory _result) public {
        articleCount++;
        articles[articleCount] = Article(_hash, _result);
    }
}
