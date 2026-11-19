from pathlib import PurePosixPath

def source_relative_path(value,dir_id):
 parts=PurePosixPath(value).parts
 if parts.count(dir_id)!=1:raise ValueError('Cannot map source path to exactly one package')
 relative=PurePosixPath(*parts[parts.index(dir_id)+1:])
 if str(relative)=='.' or '..' in relative.parts or relative.is_absolute():raise ValueError('Invalid package-relative source path')
 return relative.as_posix()
