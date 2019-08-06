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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;

import java.util.ArrayList;

/**
 * Postgres SQLDialect.
 */
public class PostgresSQLDialect extends SQLDialectBase {
    private static STGroupFile template =
        new STGroupFile("qa/qcri/nadeef/web/sql/template/PostgresTemplate.stg", '$', '$');

    /**
     * {@inheritDoc}
     */
    @Override
    public STGroupFile getTemplate() {
        return template;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String installRule() {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST instance = template.getInstanceOf("InstallRule");
        instance.add("name", "rule");
        return instance.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String installProject() {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST instance = template.getInstanceOf("InstallProject");
        instance.add("name", "project");
        return instance.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String installRuleType() {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST instance = template.getInstanceOf("InstallRuleType");
        instance.add("name", "ruletype");
        return instance.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
     public String insertProject(String projectName) {
        return
            "INSERT INTO PROJECT (dbname, name) VALUES (\'" +
                projectName +
                "\', \'" + projectName.toLowerCase() + "\')";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String queryTable(
        String tableName,
        int start,
        int interval,
        ArrayList columns,
        String filter
    ) {
        tableName = tableName.toLowerCase();
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST instance = template.getInstanceOf("QueryTable");

        instance.add("tablename", tableName);
        instance.add("start", start);
        instance.add("interval", interval);

        if (Strings.isNullOrEmpty(filter))
            instance.add("filter", "true");
        else {
            StringBuffer buf = new StringBuffer();
            int i = 0;
            for (Object column : columns) {
                if (i != 0)
                    buf.append(" or ");
                buf.append("cast(")
                    .append((String)column)
                    .append(" as text) like '%")
                    .append(filter)
                    .append("%'");
                i ++;
            }

            instance.add("filter", buf.toString());
        }

        // TODO: move this magic out
        if (tableName.equals("violation"))
            instance.add("order", "order by vid, attribute");
        else
            instance.add("order", "");

        return instance.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String querySchema(String tableName) {
        tableName = tableName.toLowerCase();
        return "SELECT * FROM " + tableName + " LIMIT 1";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String insertRule(String type, String code, String table1, String table2, String name) {
        table1 = table1.toLowerCase();
        table2 = table2.toLowerCase();
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST instance = template.getInstanceOf("InsertRule");
        instance.add("name", name);
        instance.add("type", type);
        instance.add("code", code);
        instance.add("table1", table1);
        instance.add("table2", table2);
        return instance.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String queryTopK(int k) {
        return "select tupleid, count(distinct(vid)) as count from VIOLATION group by tupleid " +
            "order by count desc LIMIT " + k;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String hasDatabase(String databaseName) {
        return "SELECT 1 FROM pg_database WHERE datname = '" + databaseName + "'";
    }
}
