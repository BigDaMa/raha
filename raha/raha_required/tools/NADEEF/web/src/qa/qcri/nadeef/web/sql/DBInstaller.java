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

package qa.qcri.nadeef.web.sql;

import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.DBMetaDataTool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.Statement;

/**
 * DB Installer for Dashboard.
 */
public class DBInstaller {
    private static Logger tracer = Logger.getLogger(DBInstaller.class);

    public static void installMetaData(DBConfig dbConfig) throws Exception {
        Connection conn = null;
        Statement stat = null;
        try {
            conn = DBConnectionPool.createConnection(dbConfig, true);
            stat = conn.createStatement();

            SQLDialect dialect = NadeefConfiguration.getDbConfig().getDialect();
            SQLDialectBase dialectInstance = SQLDialectBase.createDialectBaseInstance(dialect);

            // skip when those dbs are already existed
            DatabaseMetaData meta = conn.getMetaData();
            if (!meta.getTables(null, null, "PROJECT", null).next() &&
                !meta.getTables(null, null, "project", null).next()) {
                stat.execute(dialectInstance.installProject());
            }
        } catch (Exception ex) {
            tracer.error("Install Dashboard Meta DB failed.", ex);
            System.exit(1);
        } finally {
            if (stat != null) {
                stat.close();
            }

            if (conn != null) {
                conn.close();
            }
        }
    }

    public static void install(DBConfig dbConfig) throws Exception {
        Connection conn = null;
        Statement stat = null;
        try {
            conn = DBConnectionPool.createConnection(dbConfig, true);
            stat = conn.createStatement();

            SQLDialect dialect = dbConfig.getDialect();
            SQLDialectBase dialectInstance =
                SQLDialectBase.createDialectBaseInstance(dialect);

            // skip when those dbs are already existed
            if (!DBMetaDataTool.isTableExist(dbConfig, "RULE")) {
                stat.execute(dialectInstance.installRule());
            }

            if (!DBMetaDataTool.isTableExist(dbConfig, "RULETYPE")) {
                stat.execute(dialectInstance.installRuleType());
                stat.execute("INSERT INTO RULETYPE VALUES (0, 'UDF', true)");
                stat.execute("INSERT INTO RULETYPE VALUES (1, 'FD', true)");
                stat.execute("INSERT INTO RULETYPE VALUES (2, 'CFD', true)");
                stat.execute("INSERT INTO RULETYPE VALUES (3, 'DC', true)");
                stat.execute("INSERT INTO RULETYPE VALUES (4, 'ER', true)");
            }
        } finally {
            if (stat != null) {
                stat.close();
            }

            if (conn != null) {
                conn.close();
            }
        }
    }
}
