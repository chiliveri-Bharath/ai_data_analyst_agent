def parse_command(text):

    text = text.lower()

    if "histogram" in text:
        return "histogram"

    elif "bar chart" in text:
        return "bar"

    elif "scatter" in text:
        return "scatter"

    elif "correlation" in text:
        return "correlation"

    elif "remove outliers" in text:
        return "outliers"

    elif "dashboard" in text:
        return "dashboard"

    else:
        return "unknown"