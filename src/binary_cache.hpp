///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <sqlite3.h>
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
    class MiniDB {
    public:
        MiniDB(MiniDB const &) = delete;
        void operator=(MiniDB const &) = delete;

        MiniDB() : session_(nullptr)
        {
        }
        void open(std::string const &path)
        {
            DLPRIM_CHECK(!session_);
            if(sqlite3_open(path.c_str(),&session_)!=SQLITE_OK) {
                throw_error("Open");
            }
            sqlite3_busy_timeout(session_,1000*60); // minute timeout
        }
        ~MiniDB()
        {
            for(auto &p : cache_) {
                sqlite3_finalize(p.second);
            }
            if(session_)
                sqlite3_close(session_);
        }

        void throw_error(std::string const msg)
        {
            if(session_)
                throw ValidationError(msg + sqlite3_errmsg(session_));
            else
                throw ValidationError(msg);
        }
        void exec(char const *q)
        {
            if(sqlite3_exec(session_,q,0,0,0)!=SQLITE_OK) {
                throw_error(q);
            }
        }
        
        // it does not own statement 
        class Statement {
        public:
            Statement(sqlite3_stmt *st) : st_(st),counter_(1) {}
            Statement(Statement const &) = delete;
            void operator = (Statement const &) = delete;
            Statement(Statement &&other) 
            {
                st_=other.st_;
                counter_ = other.counter_;
                other.st_ = nullptr;
            }
            Statement &operator = (Statement &&other) 
            {
                if(this != &other) {
                    reset();
                    st_ = other.st_;
                    counter_ = other.counter_;
                    other.st_= nullptr;
                }
                return *this;
            }
            void reset()
            {
                if(st_) {
                    sqlite3_clear_bindings(st_);
                    sqlite3_reset(st_);
                    counter_ = 1;
                }
            }

            template<typename T0,typename ...Args>
            Statement &bind(T0 const &v,Args... args)
            {
                bind_val(v);
                bind(args...);
                return *this;
            }

            template<typename T0>
            Statement &bind(T0 const &v)
            {
                bind_val(v);
                return *this;
            }


            void bind_val(int64_t v)
            {
                if(sqlite3_bind_int64(st_,counter_++,v)!=SQLITE_OK)
                    throw ValidationError("Invalid numeric parameter");
            }
            void bind_val(std::string const &txt)
            {
                if(sqlite3_bind_text(st_,counter_++,txt.c_str(),txt.size(),SQLITE_TRANSIENT)!=SQLITE_OK)
                    throw ValidationError("Invalid integer parameter");
            }
            void bind_val(std::vector<unsigned char> const &blob)
            {
                if(sqlite3_bind_blob(st_,counter_++,blob.data(),blob.size(),SQLITE_TRANSIENT)!=SQLITE_OK)
                    throw ValidationError("Invalid blob parameter");
            }
            template<typename T>
            Statement &operator<<(T const &v)
            {
                bind(v);
                return *this;
            }
            bool next()
            {
                int r = sqlite3_step(st_);
                if(r == SQLITE_DONE)
                    return false;
                if(r == SQLITE_ROW)
                    return true;
                throw ValidationError("Failed to execute step");
            }
            void exec()
            {
                next();
                reset();
            }
            int64_t get_int(int index)
            {
                return sqlite3_column_int64(st_,index);
            }
            std::string get_str(int index)
            {
                char const *txt = reinterpret_cast<char const *>(sqlite3_column_text(st_,index));
                size_t len = sqlite3_column_bytes(st_,index);
                std::string r;
                r.assign(txt,len);
                return r;
            }
            void get_blob(int index,std::vector<unsigned char> &v)
            {
                unsigned const char *p = static_cast<unsigned const char *>(sqlite3_column_blob(st_,index));
                size_t len = sqlite3_column_bytes(st_,index);
                v.reserve(len);
                v.assign(p,p+len);
            }
            ~Statement()
            {
                reset();
            }
        private:
            sqlite3_stmt *st_;
            int counter_;

        };
        
        template<typename ...Args>
        Statement prepare_exec(char const *q,Args... args)
        {
            Statement st = prepare_exec(q);
            st.bind(args...);
            return st;
        }
        Statement prepare_exec(char const *q)
        {
            sqlite3_stmt *st = nullptr;
            auto p = cache_.find(q);
            if(p == cache_.end()) {
                if(sqlite3_prepare_v2(session_,q,-1,&st,0)!=SQLITE_OK) {
                    throw_error("prepare");
                }
                cache_[q] = st;
            }
            else {
                st = p->second;
            }
            return Statement(st);
        }

        class Transaction {
        public:
            Transaction(MiniDB &db) : db_(&db)
            {
                db_->exec("BEGIN;");
            }
            ~Transaction()
            {
                try {
                    rollback();
                }
                catch(...) {}
            }
            void commit()
            {
                if(db_) {
                    db_->exec("COMMIT;");
                    db_ = nullptr;
                }
            }
            void rollback()
            {
                if(db_) {
                    db_->exec("ROLLBACK;");
                    db_ = nullptr;
                }
            }
        private:
            MiniDB *db_;
        };
        
    private:
        sqlite3 *session_;
        // yes I compare addresses since all strings
        // must be static char const *
        std::map<char const *,sqlite3_stmt *> cache_;
    };

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
            
            MiniDB::Transaction tr(session_);
            auto st = session_.prepare_exec("SELECT binary,lru FROM cache WHERE key=?",key);
            if(!st.next())
                return binary;
            st.get_blob(0,binary);
            int64_t last_update = st.get_int(1);
            int64_t new_update = time(0);
            if(new_update - last_update > lru_update_)
                session_.prepare_exec("UPDATE cache SET lru=? WHERE key=?",new_update,key).exec();
            tr.commit();
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
                MiniDB::Transaction tr(session_);
                session_.prepare_exec(
                        "INSERT INTO cache(key,binary,size,lru,src,params,platform,platform_ver,device,driver_ver) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?); ",
                        key,binary,binary.size(),time(0),prog_name, params,m.platform,m.platform_ver,m.device,m.driver_ver).exec();
                session_.prepare_exec("UPDATE meta SET value=value + ? WHERE key = 'size';",binary.size()).exec();
                tr.commit();
            }
            catch(ValidationError const &e) {
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
        
            session_.open(db_path_);    
            prepare_db();
            clean_cache_overflow();
        }

        
        void prepare_db()
        {
            session_.exec(
            R"xxx(
                BEGIN;
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
                    CREATE INDEX IF NOT EXISTS cache_lru ON cache (lru);
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                    INSERT OR IGNORE INTO meta VALUES('version',1);
                    INSERT OR IGNORE INTO meta VALUES('size',0);
                    INSERT OR IGNORE INTO meta VALUES('size_limit',1073741824);
                COMMIT;
            )xxx");
        }
        void clean_cache_overflow()
        {
            MiniDB::Transaction tr(session_);
            size_t size = 0, size_limit = 0;
            auto r = session_.prepare_exec("SELECT key,value FROM meta WHERE key in ('size','size_limit')");
            while(r.next()) {
                std::string key = r.get_str(0);
                size_t val = r.get_int(1);
                if(key == "size")
                    size = val;
                else if(key == "size_limit")
                    size_limit = val;
            }
            if(size <= size_limit)
                return;
            auto q = session_.prepare_exec("SELECT size,lru FROM cache ORDER BY lru;");
            size_t remove_limit = (size - size_limit) * 3;
            size_t total = 0;
            int64_t time = -1;
            bool not_done;
            while((not_done=q.next()) == true ) {
                size_t blob_size = q.get_int(0);
                total += blob_size;
                time = q.get_int(1);
                if(total >= remove_limit)
                    break;
            }
            // make sure we compute correctly the size in case several lrus have same time stamp
            if(not_done) {
                while(q.next() && q.get_int(1) == time) {
                    total += q.get_int(0);
                }
            }
            q.reset();
            session_.prepare_exec("DELETE FROM cache WHERE lru <= ?;",time).exec();
            session_.prepare_exec("UPDATE meta SET value=value - ? WHERE key='size';",total).exec();
            tr.commit();
        }

        std::string get_path()
        {
            enable_ = true;
            lru_update_ = 3600*24; // 24h
            char *diable_cache = getenv("DLPRIM_CACHE_DISABLE");
            if(diable_cache && atoi(diable_cache) != 0) {
                enable_ = false;
                return "";
            }
            char *lru_update = getenv("DLPRIM_CACHE_LRU_UPDATE_TIMEOUT");
            if(lru_update) {
                lru_update_ = atoi(lru_update);
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

        MiniDB session_;
        std::mutex lock_;
        bool enable_;
        int64_t lru_update_;
    };
}
