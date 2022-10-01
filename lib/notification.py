import requests
from discord_webhook import DiscordWebhook


def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()


def send_discord(webhook_url, message):
    webhook = DiscordWebhook(url=webhook_url + '', content=message)
    webhook.execute()