from typing import List


class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        max_score = 0
        a = [0] * 26
        for i in range(len(letters)):
            a[ord(letters[i]) - ord('a')] += 1
        for word in words:
            tmp_score = 0
            num_net = [0] * 26
            flag = 0
            for i in range(len(word)):
                current_char = ord(word[i]) - ord('a')
                num_net[current_char] += 1
                tmp_score += score[current_char]
                if num_net[current_char] > a[current_char]:
                    flag = 1
                    break
            if flag == 0:
                if tmp_score > max_score:
                    max_score = tmp_score
        return max_score


res=Solution()
d=res.maxScoreWords(["dog","cat","dad","good"],
["a","a","c","d","d","d","g","o","o"],
[1,0,9,5,0,0,3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0])
print(d)