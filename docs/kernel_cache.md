# Kernel Cache

DLPrimitives caches kernels on non-NVidia platforms using sqlite3 cache database.
It requires dlprimitives to be compiled with libsqlite3 and the cache data base is stored
in `~/.dlprimitives/cache.db` sqlite3 file.

By default the size of cache limited by 1GB. It can be modified by updating meta table, changing
value of filed with key `size_limit` to new size in bytes, for example this way you can limit
DB size to 300MiB. Note size is approximate only.

    sqlite3 ~/.dlprimitives/cache.db
    update meta set value=300000000 where key='size_limit';


- Cache can be disabled in runtime by setting environment variable `DLPRIM_CACHE_DISABLE` to 1.
- Cache file location can be changed from default by setting environment variable `DLPRIM_CACHE_DIR`

Note: the cache is disabled on nVidia GPUs since nVidia provides its own cache of binary code. Double caching
makes it less efficient. Also what is possible to cache on nVidia platform is actually PTX "assembly" rather than 
actual binary code so you always need another level of cache PTX to Binary provided by nVidia.

