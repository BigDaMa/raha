/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.tools;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.tools.sql.SQLDialectTools;

/**
 * Configuration object for JDBC connection.
 */
public class DBConfig {
    private String userName;
    private String password;
    private String url;
    private SQLDialect dialect;

    //<editor-fold desc="Builder pattern">
    /**
     * Builder pattern to build a <code>DBConfig</code> class.
     */
    public static class Builder {
        private String userName;
        private String password;
        private String url = "localhost/unittest";
        private SQLDialect dialect = SQLDialect.DERBY;

        public Builder username(String userName) {
            this.userName = userName;
            return this;
        }

        public Builder password(String password) {
            this.password = password;
            return this;
        }

        public Builder url(String url) {
            this.url = url;
            return this;
        }

        public Builder url(String hostname, String dbname) {
            this.url(hostname + "/" + dbname);
            return this;
        }

        public Builder dialect(SQLDialect dialect) {
            this.dialect = dialect;
            return this;
        }

        public DBConfig build() {
            return new DBConfig(userName, password, url, dialect);
        }
    }
    //</editor-fold>

    //<editor-fold desc="Constructor">

    /**
     * DBConfig copy constructor.
     * @param config config.
     */
    public DBConfig(DBConfig config) {
        Preconditions.checkNotNull(config);
        this.userName = config.userName;
        this.password = config.password;
        this.url = config.url;
        this.dialect = config.dialect;
    }

    /**
     * Constructor.
     * @param userName DB user name.
     * @param password DB password.
     * @param url DB connection URL.
     * @param dialect SQL dialect.
     */
    private DBConfig(String userName, String password, String url, SQLDialect dialect) {
        Preconditions.checkArgument(
            !Strings.isNullOrEmpty(url)
        );

        this.userName = userName;
        this.password = password;
        this.url = url;
        this.dialect = dialect;
    }
    //</editor-fold>

    //<editor-fold desc="Getters">

    /**
     * Gets the user name.
     * @return user name.
     */
    public String getUserName() {
        return userName;
    }

    /**
     * Gets the password.
     * @return password.
     */
    public String getPassword() {
        return password;
    }

    /**
     * Gets the connection URL.
     * @return url.
     */
    public String getUrl() {
        if (!url.contains("jdbc:")) {
            return SQLDialectTools.buildJdbcUrl(url, dialect);
        }
        return url;
    }

    /**
     * Switches the database name.
     * @param databaseName database input.
     */
    public DBConfig switchDatabase(String databaseName) {
        if (dialect == SQLDialect.DERBYMEMORY || dialect == SQLDialect.DERBY) {
            url = getHostName() + "/" + getDatabaseName() + ";user=" + databaseName;
            userName = databaseName;
        } else {
            url = getHostName() + "/" + databaseName.toLowerCase();
        }
        return this;
    }

    /**
     * Gets the database name.
     * @return database name.
     */
    public String getDatabaseName() {
        if (url != null) {
            String[] tokens = url.split("[/;]");
            if (tokens.length >= 2) {
                return tokens[1];
            }
        }
        return null;
    }

    /**
     * Gets the host name.
     * @return host name.
     */
    public String getHostName() {
        if (url != null) {
            String[] tokens = url.split("[/;]");
            if (tokens.length >= 2) {
                return tokens[0];
            }
        }
        return "localhost";
    }

    /**
     * Gets the SQL dialect.
     * @return sql dialect.
     */
    public SQLDialect getDialect() {
        return dialect;
    }
    //</editor-fold>
}
