def split_by_re():
    import re
    def escape_special_cases(para):
        # Handle common abbreviations and titles
        abbreviations = [
            "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Capt.", "Lt.", "Col.", "Gen.",
            "Sgt.", "St.", "Mt.", "Ave.", "Rd.", "Blvd.", "Apt.", "Est.", "Jr.",
            "Sr.", "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "e.g.", "i.e.", "Nov.",
            "Dec.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sept.",
            "Oct.", "No.", "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.", "L.",
            "M.", "N.", "O.", "P.", "Q.", "R.", "S.", "T.", "U.", "V.", "W.", "X.", "Y.", "Z.",
        ]
        # Escape periods in abbreviations to avoid splitting
        for abbr in abbreviations:
            para = para.replace(abbr, abbr.replace('.', '<DOT>'))

        # Handle URLs and emails
        para = re.sub(r'(\w+://[^\s]+)', lambda m: m.group(0).replace('.', '<DOT>'), para)
        para = re.sub(r'[\w\.-]+@[\w\.-]+', lambda m: m.group(0).replace('.', '<DOT>'), para)
        para = re.sub(r'www\.[\w\.-]+', lambda m: m.group(0).replace('.', '<DOT>'), para)

        # Handle multiple periods (ellipsis)
        para = re.sub(r'(\.{3,})', lambda m: m.group(0).replace('.', '<DOT>'), para)

        # Handle decimal numbers
        para = re.sub(r'(\d)\.', r'\1<DOT>', para)
        # para = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', para)

        return para

    def restore_special_cases(para):
        # Restore periods in abbreviations, URLs, emails, ellipses, and decimal numbers
        return para.replace('<DOT>', '.')

    def split(para: str):
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 中文单字符断句符
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        # Escape special cases
        para = escape_special_cases(para)
        # 处理英文句号，排除后面紧跟两个句号的情况
        # para = re.sub(r'(?<!\.\.)\.(?!\.)([^"\'])', r".\n\1", para)
        # para = re.sub(r'([!?])([^"\'])', r"\1\n\2", para)  # 英文单字符断句符
        # para = re.sub(r'(\.{3,})([^"\'])', r"\1\n\2", para)  # 英文省略号
        para = re.sub(r'(?<!\.\.)\.(?!\.)\s*', '.\n', para)  # Periods
        para = re.sub(r'([!?])\s*', r'\1\n', para)  # Exclamation and question marks



        # Handle sentence ending with a period, question mark, or exclamation mark


        # Restore special cases
        para = restore_special_cases(para)

        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = re.sub(r'( {2,})([^”’])', r"\1\n\2", para)  # 新增的规则：将两个或多个连续的替换为换行符
        # para = re.sub(r' +', ',', para)  # 将所有连续的空格替换为逗号，而不会影响制表符或换行符。
        # para = re.sub(r' {2,}', '', para)  # 匹配两个或多个连续的空格，并将它们替换为空字符串
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    return split
