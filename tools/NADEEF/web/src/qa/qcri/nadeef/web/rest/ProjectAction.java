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
import com.google.gson.JsonObject;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.sql.DBInstaller;
import qa.qcri.nadeef.web.sql.SQLDialectBase;
import qa.qcri.nadeef.web.sql.SQLUtil;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;

import static spark.Spark.get;
import static spark.Spark.post;

public class ProjectAction {
    private static Logger tracer = Logger.getLogger(ProjectAction.class);
    private static SQLDialect dialect;
    private static SQLDialectBase dialectInstance;

    public static void setup(SQLDialect dialect_) {
        dialect = dialect_;
        dialectInstance = SQLDialectBase.createDialectBaseInstance(dialect);
        get("/project", (request, response) -> {
            String dbName = NadeefConfiguration.getDbConfig().getDatabaseName();
            return SQLUtil.query(dbName, "SELECT dbname FROM PROJECT", true);
        });

        post("/project", (request, response) -> {
            String project = request.queryParams("project");
            if (Strings.isNullOrEmpty(project) || !SQLUtil.isValidTableName(project))
                throw new IllegalArgumentException("Invalid project name " + project);
            return createProject(project);
        });
    }
    //</editor-fold>
    private static JsonObject createProject(String project) throws RuntimeException {
        // create the database
        if (dialect != SQLDialect.DERBY && dialect != SQLDialect.DERBYMEMORY) {
            // lower case for non-derby db.
            String project_ = project.toLowerCase();
            DBConfig rootConfig = new DBConfig(NadeefConfiguration.getDbConfig());
            // rootConfig.switchDatabase("");)
            try (Connection conn = DBConnectionPool.createConnection(rootConfig, true);
                 Statement stat = conn.createStatement();
                 ResultSet rs = stat.executeQuery(dialectInstance.hasDatabase(project_))) {
                if (!rs.next()) {
                    stat.execute(dialectInstance.createDatabase(project_));
                }
            } catch (Exception ex) {
                tracer.error("Create project failed " + project, ex);
                throw new RuntimeException(ex);
            }
        }

        DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
        dbConfig.switchDatabase(project);

        // install the tables
        try {
            DBInstaller.install(dbConfig);
            qa.qcri.nadeef.core.utils.sql.DBInstaller.install(dbConfig);
        } catch (Exception ex) {
            tracer.error(ex.getMessage(), ex);
            throw new RuntimeException(ex);
        }

        // TODO: magic string, and missing transaction.
        return SQLUtil.update(
            NadeefConfiguration.getDbConfig().getDatabaseName(),
            dialectInstance.insertProject(project)
        );
    }
}
