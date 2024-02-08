from langchain.tools import BaseTool


class StringReverseTool(BaseTool):
    name = "String Reversal Tool"
    description = "use this tool when you need to reverse a strng"


def _run(self, word: str):
    return word[::-1]


def _arun(self, word: str):
    raise NotImplementedError("Async not supported by StringReverseTool")