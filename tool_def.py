"""Module containing custom YF tools to be used eventually by LangGraph Agent"""
from langchain_core.tools import tool, StructuredTool
import yfinance as yf

@tool
def company_address(ticker: str) -> str:
    """
    Generates company address from the stock ticker provided.

    Args:
        ticker (str): Stock ticker of company.

    Returns:
        str: Address of relevant company.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return " ".join([info[key] for key in ["address1", "city", "state", "zip", "country"]])

@tool
def fulltime_employees(ticker: str)-> int:
    """
    Generates number of full time employees at company from the stock ticker provided.

    Args:
        ticker (str): Stock ticker of company.

    Returns:
        int: Number of full time employees at company.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info["fullTimeEmployees"]

@tool
def last_close_price(ticker: str)-> float:
    """
    Generates the last closing price from the stock ticker provided.

    Args:
        ticker (str): Stock ticker of company.

    Returns:
        float: Last closing price of stock of company.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info["previousClose"]

@tool
def ebitda(ticker: str)-> float:
    """
    Generates EBITDA of company from stock ticker provided.

    Args:
        ticker (str): Stock ticker of company.

    Returns:
        float: EBITDA of company.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.get_info()
    return info["ebitda"]

tools = [
    company_address,
    fulltime_employees,
    last_close_price,
    ebitda
]