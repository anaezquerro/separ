import requests

def send_telegram(message):
    apiToken = '5909368932:AAHUIpRsSOy717DgWXt1KpHCJhj-vxs-9JU'
    chatID = '909012705'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        requests.post(apiURL, json={'chat_id': chatID, 'text': message})
    except Exception as e:
        pass