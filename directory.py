#Directory Path

def path():

    OS = 'Raspberry Flask'

    if OS == 'Windows':
        directory_path = "F:/EGH400 Proj/main/"
    elif OS == 'Mac':
        directory_path = "/Volumes/Antony HHD/EGH400 Proj/main/"
    elif OS == 'Mac Flask':
        directory_path = "/Users/Antony_/webapp/" 
    elif OS == 'Raspberry':
        directory_path = "/media/pi/Antony HHD/EGH400 Proj/main/" 
    elif OS == 'Raspberry Flask':
        directory_path = "/webapp/"

    return directory_path

