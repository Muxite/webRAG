from bs4 import BeautifulSoup

# TODO benchmark BeautifulSoup vs regex vs lxml_html_clean
def clean_operation(html):
    """
    Extract simplified main text content from the provided HTML.

    Include links in the output but exclude images.
    """
    soup = BeautifulSoup(html, "html.parser")

    for unwanted_tag in soup(["script", "style"]):
        unwanted_tag.decompose()

    main_text = soup.get_text(separator="\n", strip=True)
    return main_text