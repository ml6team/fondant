import gzip
import os
import tarfile
import zipfile
from io import BytesIO

import dask
import fsspec
import pytest
from fsspec.spec import AbstractFileSystem

from src.main import (
    AbstractFileHandler,
    DirectoryHandler,
    FileHandler,
    FilesToDaskConverter,
    GzipFileHandler,
    TarFileHandler,
    ZipFileHandler,
    get_file_handler,
    get_filesystem,
)


class TestFileHandler:
    @pytest.fixture(autouse=True)
    def __setup_and_teardown(self):
        """Create dummy file before test and remove it after."""
        self.file_content = "This is a test content"
        self.file_name = "test.txt"
        self.file_path = "/tmp/" + self.file_name

        with open(self.file_path, "w") as f:
            f.write(self.file_content)

        self.file_handler = FileHandler(self.file_path)
        yield
        os.remove(self.file_path)

    def test_read(self):
        """Test the read method."""
        file_name, file_content = next(self.file_handler.read())
        assert isinstance(file_name, str)
        assert isinstance(file_content, BytesIO)
        result = (file_name, file_content.getvalue())
        expected_result = (
            self.file_name,
            BytesIO(self.file_content.encode()).getvalue(),
        )

        assert result == expected_result


class TestGzipFileHandler:
    @pytest.fixture(autouse=True)
    def __setup_and_teardown(self):
        print("Setup...")
        self.filename = "test.gz"
        self.filepath = "/tmp/" + self.filename
        self.file_content = b"Some test content"
        with gzip.open(self.filepath, "wb") as f:
            f.write(self.file_content)

        self.handler = GzipFileHandler(self.filepath)
        yield
        print("Teardown...")
        os.remove(self.filepath)

    def test_read(self):
        result = list(self.handler.read())
        assert len(result) > 0, "No data read from gzip file"

        filename, content = result[0]
        assert isinstance(filename, str), "Filename should be a string"
        assert isinstance(content, BytesIO), "Content should be a BytesIO object"

        assert filename == self.filename
        assert content.getvalue() == self.file_content


class TestZipFileHandler:
    @pytest.fixture(autouse=True)
    def __setup_method(self):
        print("Setting up...")
        self.filepath = "test.zip"

        with zipfile.ZipFile(self.filepath, "w") as zipf:
            zipf.writestr("test1.txt", b"some content")

        self.handler = ZipFileHandler(self.filepath)

        yield
        print("Tearing down...")
        os.remove(self.filepath)

    def test_read_normal_zipfile(self):
        filename, content = next(self.handler.read())
        assert isinstance(filename, str), "Filename should be a string"
        assert isinstance(content, BytesIO), "Content should be a BytesIO object"

        assert filename == "test1.txt"
        assert content.read() == b"some content"


class TestTarFileHandler:
    @pytest.fixture(autouse=True)
    def __setup_method(self):
        print("Setting up...")
        self.filepath = "test.tar.gz"

        with tarfile.open(self.filepath, "w:gz") as tarf:
            data = b"some content"
            info = tarfile.TarInfo(name="test1.txt")
            info.size = len(data)
            tarf.addfile(info, BytesIO(data))

        self.handler = TarFileHandler(self.filepath)

        yield
        print("Tearing down...")
        os.remove(self.filepath)

    def test_read_normal_tarfile(self):
        filename, content = next(self.handler.read())
        assert isinstance(filename, str), "Filename should be a string"
        assert isinstance(content, BytesIO), "Content should be a BytesIO object"

        assert filename == "test1.txt"
        assert content.read() == b"some content"


class TestDirectoryHandler:
    @pytest.fixture(autouse=True)
    def __setup_method(self, tmpdir):
        print("Setting up...")
        self.num_files = 4
        # Setting up some test data
        self.test_dir = str(tmpdir.mkdir("sub"))

        # Create normal text file
        self.filepath = os.path.join(self.test_dir, "test1.txt")
        with open(self.filepath, "w") as f:
            f.write("some content")

        # Create gzipped file
        self.gz_filepath = os.path.join(self.test_dir, "test2.txt.gz")
        with gzip.open(self.gz_filepath, "wb") as f:
            f.write(b"some gzipped content")

        # Create zip file
        self.zip_filepath = os.path.join(self.test_dir, "test3.zip")
        with zipfile.ZipFile(self.zip_filepath, "w") as zp:
            zp.writestr("text_file_inside.zip", "zip content")

        # Create tar file
        self.tar_filepath = os.path.join(self.test_dir, "test4.tar")
        with tarfile.open(self.tar_filepath, "w") as tp:
            tp.add(self.filepath, arcname="text_file_inside.tar")

        self.directory_handler = DirectoryHandler(self.test_dir)

    def test_read_directory_with_compressed_files(self):
        file_list = list(self.directory_handler.read())

        assert len(file_list) == self.num_files, "Four Files should be read"
        print("filelist: ", file_list)
        txt_file = next((f for f in file_list if f[0] == "test1.txt"), None)
        gz_file = next((f for f in file_list if f[0] == "test2.txt.gz"), None)
        zip_file = next((f for f in file_list if f[0] == "text_file_inside.zip"), None)
        tar_file = next((f for f in file_list if f[0] == "text_file_inside.tar"), None)

        assert txt_file is not None, "txt file not read correctly"
        assert gz_file is not None, "gz file not read correctly"
        assert zip_file is not None, "zip file not read correctly"
        assert tar_file is not None, "tar file not read correctly"

        # Further assertions to verify the content of the read files
        with open(self.filepath) as f:
            assert f.read() == txt_file[1].getvalue().decode(
                "utf-8",
            ), "Text file content does not match"

        with gzip.open(self.gz_filepath, "rb") as f:
            assert (
                f.read() == gz_file[1].getvalue()
            ), "Gzipped file content does not match"

        with zipfile.ZipFile(self.zip_filepath, "r") as zp:
            assert (
                zp.read("text_file_inside.zip") == zip_file[1].getvalue()
            ), "Zip file content does not match"

        with tarfile.open(self.tar_filepath, "r") as tp:
            assert (
                tp.extractfile("text_file_inside.tar").read() == tar_file[1].getvalue()
            ), "Tar file content does not match"


def test_get_file_handler():
    fs = fsspec.filesystem("file")  # setting up a local file system

    # Test case 1: For .gz file
    handler = get_file_handler("testfile.gz", fs)
    assert isinstance(
        handler,
        GzipFileHandler,
    ), "For .gz file, the handler should be of type GzipFileHandler."

    # Test case 2: For .tar file
    handler = get_file_handler("testfile.tar", fs)
    assert isinstance(
        handler,
        TarFileHandler,
    ), "For .tar file, the handler should be of type TarFileHandler."

    # Test case 3: For .tar.gz file
    handler = get_file_handler("testfile.tar.gz", fs)
    assert isinstance(
        handler,
        TarFileHandler,
    ), "For .tar.gz file, the handler should be of type TarFileHandler."

    # Test case 4: For .zip file
    handler = get_file_handler("testfile.zip", fs)
    assert isinstance(
        handler,
        ZipFileHandler,
    ), "For .zip file, the handler should be of type ZipFileHandler."

    # Test case 5: For unsupported file type
    handler = get_file_handler("testfile.txt", fs)
    assert isinstance(
        handler,
        DirectoryHandler,
    ), "For unsupported file types, the handler should default to DirectoryHandler."


def test_files_to_dask_converter():
    # Use a mock file handler
    class MockFileHandler(AbstractFileHandler):
        def read(self):
            yield "file1", BytesIO(b"content1")
            yield "file2", BytesIO(b"content2")

    # Test case 1: Testing FilesToDaskConverter with two simple files
    handler = MockFileHandler("/path", fsspec.filesystem("file"))
    converter = FilesToDaskConverter(handler)
    ddf = converter.to_dask_dataframe(chunksize=2)
    num_records = 2
    assert (
        ddf.npartitions == num_records
    ), "Number of partitions should match chunksize."

    result = dask.compute(ddf)[0]

    assert len(result) == num_records, "The dataframe should have content of two files."
    assert list(result.filename) == [
        "file1",
        "file2",
    ], "The dataframe index should include the filenames."
    assert list(result.content) == [
        b"content1",
        b"content2",
    ], "The dataframe Content column should include the file contents."


def test_get_filesystem_local(mocker):
    """Test local file system."""
    mock_fs = mocker.patch("fsspec.filesystem", return_value=AbstractFileSystem())
    fs = get_filesystem("file:///path/to/file")
    assert isinstance(fs, AbstractFileSystem)
    mock_fs.assert_called_once_with("file")


def test_get_filesystem_s3(mocker):
    """Test S3 file system."""
    mock_fs = mocker.patch("fsspec.filesystem", return_value=AbstractFileSystem())
    fs = get_filesystem("s3://bucket/key")
    assert isinstance(fs, AbstractFileSystem)
    mock_fs.assert_called_once_with("s3")


def test_get_filesystem_gcs(mocker):
    """Test Google Cloud Storage file system."""
    mock_fs = mocker.patch("fsspec.filesystem", return_value=AbstractFileSystem())
    fs = get_filesystem("gs://bucket/key")
    assert isinstance(fs, AbstractFileSystem)
    mock_fs.assert_called_once_with("gcs")


def test_get_filesystem_abfs(mocker):
    """Test Azure Blob Storage file system."""
    mock_fs = mocker.patch("fsspec.filesystem", return_value=AbstractFileSystem())
    fs = get_filesystem("abfs://container/path")
    assert isinstance(fs, AbstractFileSystem)
    mock_fs.assert_called_once_with("abfs")


def test_get_filesystem_unsupported_scheme(mocker):
    """Test unsupported scheme."""
    mocker.patch("fsspec.filesystem", return_value=AbstractFileSystem())
    fs = get_filesystem("unsupported://bucket/key")
    assert fs is None
