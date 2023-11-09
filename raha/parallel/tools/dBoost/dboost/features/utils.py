import unicodedata

def string_normalize(s): # http://stackoverflow.com/questions/517923/
   return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
