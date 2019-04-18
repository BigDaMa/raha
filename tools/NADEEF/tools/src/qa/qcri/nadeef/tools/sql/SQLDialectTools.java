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

package qa.qcri.nadeef.tools.sql;

/**
 * Base class for cross vendor Database calling.
 */
public abstract class SQLDialectTools {
    /**
     * Builds up JDBC connection url.
     * @param url url.
     * @param dialect sql dialect.
     * @return JDBC connection url string.
     */
    public static String buildJdbcUrl(String url, SQLDialect dialect) {
        StringBuilder jdbcUrl = new StringBuilder("jdbc:");
        switch (dialect) {
            default:
            case DERBY:
                jdbcUrl.append("derby://").append(url);
                break;
            case DERBYMEMORY:
                jdbcUrl.append("derby:").append(url);
                break;
            case POSTGRES:
                jdbcUrl.append("postgresql://").append(url);
                break;
            case MYSQL:
                jdbcUrl.append("mysql://").append(url);
                break;
        }

        return jdbcUrl.toString();
    }

    /**
     * Gets the {@link SQLDialect} from a string.
     * @param type type string.
     * @return sql dialect.
     */
    public static SQLDialect getSQLDialect(String type) {
        SQLDialect result;
        switch (type) {
            default:
            case "derby":
                result = SQLDialect.DERBY;
                break;
            case "postgres":
                result = SQLDialect.POSTGRES;
                break;
            case "mysql":
                result = SQLDialect.MYSQL;
                break;
            case "derbymemory":
                result = SQLDialect.DERBYMEMORY;
        }
        return result;
    }

    public static String getDriverName(SQLDialect dialect) {
        switch (dialect) {
            case POSTGRES:
                return "org.postgresql.Driver";
            case DERBYMEMORY:
                return "org.apache.derby.jdbc.EmbeddedDriver";
            case DERBY:
                return "org.apache.derby.jdbc.ClientDriver";
            case MYSQL:
                return "com.mysql.jdbc.Driver";
            default:
                throw new UnsupportedOperationException();
        }
    }
}
