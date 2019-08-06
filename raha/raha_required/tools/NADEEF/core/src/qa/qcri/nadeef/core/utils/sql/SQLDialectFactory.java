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

package qa.qcri.nadeef.core.utils.sql;

import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.tools.sql.SQLDialect;

/**
 * Factory class to generate {@link SQLDialectBase} class instance.
 */
public class SQLDialectFactory {

    /**
     * Returns instance of dialect manager based on input dialect.
     * @param dialect input dialect.
     * @return dialect manager instance.
     */
    public static SQLDialectBase getDialectManagerInstance(SQLDialect dialect) {
        SQLDialectBase result = null;
        switch (dialect) {
            case DERBYMEMORY:
            case DERBY:
                result = new DerbySQLDialect();
                break;
            case POSTGRES:
                result = new PostgresSQLDialect();
                break;
            case MYSQL:
                result = new MySQLDialect();
                break;
        }
        return result;
    }

    /**
     * Returns NADEEF dialect manager instance.
     * @return dialect manager instance.
     */
    public static SQLDialectBase getNadeefDialectManagerInstance() {
        SQLDialect dialect = NadeefConfiguration.getDbConfig().getDialect();
        return getDialectManagerInstance(dialect);
    }
}
