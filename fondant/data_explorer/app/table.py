from render import image_renderer, load_image, make_render_image_template


def convert_image_column(dataframe, field):
    dataframe[field] = dataframe[field].apply(load_image)
    dataframe[field] = dataframe[field].apply(image_renderer)
    return dataframe


def configure_image_builder(builder, field):
    render_template = make_render_image_template(field)
    builder.configure_column(field, field, cellRenderer=render_template)


def get_image_fields(dataframe):
    # check which of the columns contain byte data
    image_fields = []
    for field in dataframe.columns:
        print(field, dataframe[field].dtype)
        if dataframe[field].dtype == "object":
            image_fields.append(field)
    return image_fields


def get_numeric_fields(dataframe):
    # check which of the columns contain byte data
    numeric_fields = []
    for field in dataframe.columns:
        if dataframe[field].dtype in ["int16", "int32", "float16", "float32"]:
            numeric_fields.append(field)
    return numeric_fields
