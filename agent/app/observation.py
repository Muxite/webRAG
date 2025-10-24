from trafilatura import extract

def clean_operation(html):
    main_text = extract(html, include_links=True, include_images=False)
    return main_text