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

import com.google.common.base.Preconditions;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * Database manager for Apache Derby database.
 */
public class MySQLDialect extends SQLDialectBase {
    public static STGroupFile template =
        new STGroupFile(
            "qa*qcri*nadeef*core*utils*sql*template*MySQLTemplate.stg".replace(
                "*", "/"), '$', '$');
    /**
     * {@inheritDoc}
     */
    @Override
    protected STGroupFile getTemplate() {
        return template;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void copyTable(Connection conn, String sourceName, String targetName)
            throws SQLException {
        Statement stat = null;
        try {
            stat = conn.createStatement();
            stat.execute("CREATE TABLE " + targetName + " AS SELECT * FROM " + sourceName);
            ResultSet rs = stat.executeQuery(
                "select * from information_schema.columns where table_name = '" +
                    targetName + "' and column_name = 'tid'"
            );

            if (!rs.next()) {
                stat.execute("alter table " + targetName + " add column tid serial primary key");
            }
        } finally {
            if (stat != null) {
                stat.close();
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String dropIndex(String indexName, String tableName) {
        return "DROP INDEX " + indexName +  " ON " + tableName;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String countTable(String tableName) {
        ST st = getTemplate().getInstanceOf("CountTable");
        st.add("tableName", tableName);
        return st.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String limitRow(int row) {
        return " LIMIT " + row;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String createTableFromCSV(String tableName, String content) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());

        StringBuilder sqlBuilder = new StringBuilder(1024);
        String[] tokens = content.split(",");
        int i = 0;
        for (String token : tokens) {
            if (i != 0) {
                sqlBuilder.append(", ");
            }
            String[] vs = token.trim().split("\\s");
            sqlBuilder.append("`").append(vs[0]).append("` ").append(vs[1]);
            i ++;
        }

        ST st = template.getInstanceOf("CreateTableFromCSV");
        st.add("tableName", tableName);
        st.add("content", sqlBuilder.toString());
        return st.render();
    }
}
