def autoconv(field, floats_only = False):
    converters = [float] if floats_only else [int, float]

    for conv in converters:
        try:
            return conv(field)
        except ValueError:
            pass

    return field
