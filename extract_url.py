from youtube_dl import YoutubeDL


def sys_main():
    ydl = YoutubeDL()
    ydl.add_default_info_extractors()
    info = ydl.extract_info('https://www.youtube.com/watch?v=-tzvebu6U08', download=False)
    for x in info['formats']:
        print(x['format'], x['url'])
    return 0

if __name__ == '__main__':
    sys_main()
