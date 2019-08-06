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
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBMetaDataTool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.util.List;

import static spark.Spark.get;

public class SourceAction {
    private static final String TABLE_PREFIX = "TB_"; // TODO: move

    //<editor-fold desc="source actions">
    public static void setup(SQLDialect dialect) {
        Logger tracer = Logger.getLogger(SourceAction.class);

        get("/:project/data/source", (request, response) -> {
            JsonObject json = new JsonObject();
            JsonArray result = new JsonArray();
            String project = request.params("project");

            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
            dbConfig.switchDatabase(project);

            try {
                List<String> tables = DBMetaDataTool.getTables(dbConfig);
                for (String tableName : tables)
                    if (!tableName.equalsIgnoreCase("AUDIT") &&
                        !tableName.equalsIgnoreCase("VIOLATION") &&
                        !tableName.equalsIgnoreCase("RULE") &&
                        !tableName.equalsIgnoreCase("RULETYPE") &&
                        !tableName.equalsIgnoreCase("REPAIR") &&
                        !tableName.equalsIgnoreCase("PROJECT") &&
                        tableName.toUpperCase().startsWith(TABLE_PREFIX))
                        result.add(new JsonPrimitive(tableName));
                json.add("data", result);
                return json;
            } catch (Exception ex) {
                tracer.error("Exception", ex);
                throw new RuntimeException(ex);
            }
        });
    }
    //</editor-fold>
}
