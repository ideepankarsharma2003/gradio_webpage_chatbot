import requests
from bs4 import BeautifulSoup


def clean(url,  tries=0):
    print(f'url: {url}')
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unnecessary elements (styles, scripts, etc.)
        for script in soup(['script', 'style', 'a', 'img', 'video']):
            script.extract()
        
        cleaned_text = soup.get_text()
        cleaned_text= cleaned_text.strip()
        cleaned_text= cleaned_text.replace('\n', ' ')
        cleaned_text= cleaned_text.replace('\t', ' ')
        print("success 😁")
        
        return cleaned_text.replace('    ', '')
            
    else:
        print ("Failed to retrieve the webpage: ", url)
        if tries==3:
            print("Tried 3 times , skipping url: ", url)
            return ''
        print("Retrying over scrapper....")
        if tries>0:
            return extract_paragraphs(url, tries+1)
        return clean("http://api.scraperapi.com?api_key=f86bd0a9f5e74a8b616c494a7682f4d2&url="+url, tries+1)




def extract_paragraphs(url, tries=0):
    print(f"url: {url}")
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find and extract all paragraphs (usually enclosed in <p> tags)
            paragraphs = soup.find_all('p')

            # Extract the text from each paragraph
            paragraph_texts = [p.get_text() for p in paragraphs]
            print("success 😁")
            return paragraph_texts

        else:
            print(f"Failed to fetch content. Status code: {response.status_code} for url: {url}")
            if tries==3:
                print("Tried 3 times , skipping url: ", url)                
                return ''
            print("Retrying over scrapper....")
            if tries>0:
                return extract_paragraphs(url, tries+1)
                
            return extract_paragraphs("http://api.scraperapi.com?api_key=f86bd0a9f5e74a8b616c494a7682f4d2&url="+url, tries+1)
            
    except Exception as e:
        print(f"An error occurred: {str(e)} for url: {url}")

