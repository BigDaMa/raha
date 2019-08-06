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

package qa.qcri.nadeef.web.rest;

import com.google.common.base.Strings;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.HTTPPostJsonParser;
import qa.qcri.nadeef.web.sql.SQLDialectBase;
import qa.qcri.nadeef.web.sql.SQLUtil;

import java.util.HashMap;
import java.util.List;

import static spark.Spark.*;

public class RuleAction {

    //<editor-fold desc="Rule actions">
    public static void setup(SQLDialect dialect) {
        SQLDialectBase dialectInstance = SQLDialectBase.createDialectBaseInstance(dialect);
        get("/:project/data/rule", (x, response) -> {
            String project = x.params("project");
            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            return SQLUtil.query(project, dialectInstance.queryRule(), true);
        });

        get("/:project/data/rule/:ruleName", (x, response) -> {
            String ruleName = x.params("ruleName");
            String project = x.params("project");
            if (Strings.isNullOrEmpty(project) || Strings.isNullOrEmpty(ruleName))
                throw new IllegalArgumentException("Input is not valid.");

            return SQLUtil.query(project, dialectInstance.queryRule(ruleName), true);
        });

        delete("/:project/data/rule", (x, response) -> {
            HashMap<String, Object> json = HTTPPostJsonParser.parse(x.body());
            @SuppressWarnings("unchecked")
            List<String> ruleNames = (List<String>)json.get("rules");
            String project = (String)json.get("project");

            if (Strings.isNullOrEmpty(project) ||
                ruleNames == null ||
                ruleNames.size() == 0)
                throw new IllegalArgumentException("Input is not valid.");

            for (String ruleName : ruleNames)
                SQLUtil.update(project, dialectInstance.deleteRule(ruleName));
            return 0;
        });

        post("/:project/data/rule", (request, response) -> {
            String project = request.params("project");
            String type = request.queryParams("type");
            String name = request.queryParams("name");
            String table1 = request.queryParams("table1");
            String table2 = request.queryParams("table2");
            String code = request.queryParams("code");
            if (Strings.isNullOrEmpty(type)
                || Strings.isNullOrEmpty(name)
                || Strings.isNullOrEmpty(table1)
                || Strings.isNullOrEmpty(code)
                || Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            // Doing a delete and insert
            SQLUtil.update(project, dialectInstance.deleteRule(name));

            return SQLUtil.update(
                project,
                dialectInstance.insertRule(type.toUpperCase(), code, table1, table2, name)
            );
        });
    }
    //</editor-fold>
}
