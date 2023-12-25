# from pathlib import Path

# Path(r"C:\Prgram Files\Microsoft")
# Path("/usr/local/bin")
# Path() / "ecommerce" / "__init__py"
# Path.home()

# path = Path("ecommerce/__init__.py")
# path.exists()
# path.is_file()
# path.is_dir()
# path.mkdir()
# path.rmdir()
# path.rename("ecommerce2")
# print(path.name)
# print(path.stem)
# print(path.suffix)
# print(path.parent)
# path = path.with_name("file.txt")
# print(path.absolute())
# path = path.with_suffix(".txt")
# print(path)
from pathlib import Path
from time import ctime
import shutil

source = Path("mosh/ecommerce/__init__.py")
target = Path() / "__init__.py"
shutil.copy(source, target)  # 复制文件
# path = Path("mosh/ecommerce")

# paths = [p for p in path.iterdir() if p.is_dir()]
# print(paths)
# # py_files = [p for p in path.glob("*.py")]
# py_files = [p for p in path.rglob("*.py")]
# print(py_files)
# path = Path("mosh/ecommerce/__init__.py")
# path.rename("init.txt")
# path.unlink()
# print(ctime(path.stat().st_ctime))
# print(path.read_text())
# path.write_text("...")
