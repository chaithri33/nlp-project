import requests
from bs4 import BeautifulSoup
import json

def extract_title_and_paragraphs(url):
   
    try:
        response = requests.get(url)
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_element = soup.find('h1')
        page_title = title_element.get_text().strip() if title_element else 'No title found'
        
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
        content = ' '.join(paragraphs)

        output_data = {
            'title': page_title,
            'content': content
        }
        
        output_filename = 'scraped_data.json'
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, ensure_ascii=False, indent=4)
        
        print(f"Data successfully saved to '{output_filename}'")
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the webpage: {e}")

url = 'https://www.bbc.com/telugu/india-50715656'
extract_title_and_paragraphs(url)
