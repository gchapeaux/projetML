def download_from_net(imgid):
    page = urllib.request.urlopen(f'http://visualgenome.org/api/v0/images/{imgid}?format=json')
    img_url = json.load(page)['url']
    img = urllib.request.urlopen(img_url)
    extension = img_url.split(".")[-1]
    file = open(f'dataset/{imgid}.{extension}','wb')
    file.write(img.read())
    file.close()

if __name__ == "__main__":
    for imgid in range(1000):
        download_from_net(imgid)