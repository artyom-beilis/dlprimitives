#pragma once

#include <cppdb/frontend.h>
#include <thread>
#include <sstream>
#include "sha1.hpp"
#ifdef DLPRIM_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace dlprim {
    class BinaryProgramCache {
    public:
        struct Meta {
            std::string platform, platform_ver, device, vendor, driver_ver;
            std::string to_str() const
            {
                return platform + "\n" + platform_ver + "\n" + device + "\n" + vendor + "\n" + driver_ver+"\n";
            }
        };

        static BinaryProgramCache &instance()
        {
            static std::unique_ptr<BinaryProgramCache> cache;
            static std::once_flag once;
            std::call_once(once,init,cache);
            return *cache;
        }

        std::vector<unsigned char> get_binary(Context &ctx,std::string const &source,std::string const &params)
        {
            std::vector<unsigned char> binary;
            if(!enable_)
                return binary;

            std::unique_lock<std::mutex> g(lock_);
            Meta m = get_meta(ctx);
            std::string key = get_key(m,source,params);
            
            cppdb::transaction tr(session_);
            cppdb::result res = session_ << "SELECT binary FROM cache WHERE key=?" << key << cppdb::row;
            if(res.empty())
                return binary;
            std::string binary_str = res.get<std::string>(0);
            session_ << "UPDATE cache SET lru=? WHERE key=?" << time(0) << key << cppdb::exec;
            tr.commit();
            binary.assign(binary_str.begin(),binary_str.end());
            return binary;
        }

        void save_binary(Context &ctx,std::string const &source,std::string const &params,std::vector<unsigned char> const &binary,std::string const &prog_name)
        {
            if(binary.empty() || !enable_)
                return;
            std::unique_lock<std::mutex> g(lock_);

            Meta m = get_meta(ctx);
            std::string key = get_key(m,source,params);
            std::stringstream ss;
            ss.write((char *)binary.data(),binary.size());
            ss.seekg(0);
            try {
                cppdb::transaction tr(session_);
                session_ << "INSERT INTO cache(key,binary,size,lru,src,params,platform,platform_ver,device,driver_ver) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?); "
                     << key << static_cast<std::istream &>(ss) << binary.size() << time(0) << prog_name << params <<m.platform << m.platform_ver<< m.device << m.driver_ver << cppdb::exec;
                session_ << "UPDATE meta SET value=value + ? WHERE key = 'size';" << binary.size() << cppdb::exec;
                session_.commit();
            }
            catch(cppdb::cppdb_error const &e) {
                std::cerr << e.what() << std::endl;
                // There may be a error due to collision
                // ignore
            }
        }

        bool enabled()
        {
            return enable_;
        }

    private:
        static void init(std::unique_ptr<BinaryProgramCache> &cache)
        {
            cache.reset(new BinaryProgramCache());
        }

        BinaryProgramCache() 
        {
            std::string path_ = get_path();
            if(!enable_)
                return;
            std::string db_path_ = path_ + "/cache.db";
            #ifdef DLPRIM_WINDOWS
            _mkdir(path_.c_str());
            #else
            mkdir(path_.c_str(),0777);
            #endif
        
            session_.open("sqlite3:db=" + db_path_);    
            prepare_db();
        }

        
        void prepare_db()
        {
            cppdb::transaction tr(session_);
            session_ << 
            R"xxx(
                    CREATE TABLE IF NOT EXISTS cache (
                    	key TEXT PRIMARY KEY,
                        binary BLOB NOT NULL,
                        size INTEGER NOT NULL,
                        lru INTEGER NOT NULL, 
                        src TEXT NOT NULL default '',
                        params TEXT NOT NULL default '',
                        platform TEXT NOT NULL default '',
                        platform_ver TEXT NOT NULL default '',
                        device TEXT NOT NULL default '',
                        driver_ver TEXT NOT NULL default ''
                    );
            )xxx" << cppdb::exec;
            session_ << "CREATE INDEX IF NOT EXISTS cache_lru ON cache (lru);" << cppdb::exec;
            session_ << R"xxx(
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                )xxx" << cppdb::exec;

            session_ << "INSERT OR IGNORE INTO meta VALUES('version',1);" << cppdb::exec;
            session_ << "INSERT OR IGNORE INTO meta VALUES('size',0);" << cppdb::exec;
            // default limit is 1G
            session_ << "INSERT OR IGNORE INTO meta VALUES('size_limit',1073741824);" << cppdb::exec; 
            tr.commit();
        }

        std::string get_path()
        {
            enable_ = true;
            char *diable_cache = getenv("DLPRIM_CACHE_DISABLE");
            if(diable_cache && atoi(diable_cache) != 0) {
                enable_ = false;
                return "";
            }
            char *cache_dir = getenv("DLPRIM_CACHE_DIR");
            if(cache_dir)
                return cache_dir;
            #ifndef DLPRIM_WINDOWS
            char *home = getenv("HOME");
            if(home) {
                return std::string(home) + "/.dlprimitives";
            }
            #else
            char *local_app_data = getenv("LOCALAPPDATA");
            if(local_app_data) {
                return std::string(local_app_data) + "\\dlprimitives";
            }
            #endif
            enable_ = false;
            return "";
        }

        
       
        static Meta get_meta(Context &ctx)
        {
            Meta meta;
            // c_str because some drivers return buggy strings with extra 0
            meta.platform = ctx.platform().getInfo<CL_PLATFORM_NAME>().c_str();
            meta.platform_ver = ctx.platform().getInfo<CL_PLATFORM_VERSION>().c_str();
            meta.device = ctx.device().getInfo<CL_DEVICE_NAME>().c_str();
            meta.vendor = ctx.device().getInfo<CL_DEVICE_VENDOR>().c_str();
            meta.driver_ver = ctx.device().getInfo<CL_DRIVER_VERSION>().c_str();
            return meta;
        }
        std::string get_key(Meta const &meta,std::string const &source,std::string const &params)
        {
            sha1 s;
            std::string smeta = meta.to_str();
            std::string sep = "---ba37c4e556d2fdc0f5acc289049b17ef--\n";
            s.process_bytes(smeta.c_str(),smeta.size());
            s.process_bytes(sep.c_str(),sep.size());
            s.process_bytes(source.c_str(),source.size());
            s.process_bytes(sep.c_str(),sep.size());
            s.process_bytes(params.c_str(),params.size());
            union {
                unsigned int digest[5];
                unsigned char bdigest[20];
            } dg;
            s.get_digest(dg.digest);
            unsigned char *p=dg.bdigest;
            std::string res;
            for(size_t i=0;i<20;i++) {
                char const *h="0123456789abcdef";
                res += h[(p[i]>>4)&0xF];
                res += h[p[i] & 0xF];
            }
            return res;
        }

        cppdb::session session_;
        std::mutex lock_;
        bool enable_;
    };
}
