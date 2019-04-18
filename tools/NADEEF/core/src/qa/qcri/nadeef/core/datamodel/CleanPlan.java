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

package qa.qcri.nadeef.core.datamodel;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Files;
import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.core.utils.sql.DBMetaDataTool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.tools.sql.SQLDialectTools;

import java.io.File;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * NADEEF cleaning plan.
 */
public class CleanPlan {
    private DBConfig source;
    private Rule rule;
    private static Logger tracer = Logger.getLogger(CleanPlan.class);

    // <editor-fold desc="Constructor">
    /**
     * Constructor.
     */
    public CleanPlan(DBConfig sourceConfig, Rule rule) {
        this.source = sourceConfig;
        this.rule = rule;
    }

    // </editor-fold>

    /**
     * Creates a {@link CleanPlan} from JSON object.
     *
     * @param reader JSON string reader.
     * @param dbConfig Nadeef DB config.
     * @return instance.
     */
    public static List<CleanPlan> create(Reader reader, DBConfig dbConfig) throws Exception {
        Preconditions.checkNotNull(reader);
        GsonBuilder gson = new GsonBuilder();
        gson.registerTypeAdapter(CleanPlanJsonAdapter.class, new CleanPlanJsonDeserializer());
        gson.setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES);
        CleanPlanJsonAdapter adapter =
            gson.create().fromJson(reader, CleanPlanJsonAdapter.class);
        return create(adapter, dbConfig);
    }

    /**
     * Creates a {@link CleanPlan} from JSON object.
     *
     * @param json JSON object.
     * @param dbConfig Nadeef DB config.
     * @return instance.
     */
    public static List<CleanPlan> create(JsonObject json, DBConfig dbConfig) throws Exception {
        Preconditions.checkNotNull(json);
        GsonBuilder gson = new GsonBuilder();
        gson.registerTypeAdapter(CleanPlanJsonAdapter.class, new CleanPlanJsonDeserializer());
        gson.setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES);
        CleanPlanJsonAdapter adapter =
            gson.create().fromJson(json, CleanPlanJsonAdapter.class);
        return create(adapter, dbConfig);
    }

    // <editor-fold desc="Property Getters">

    /**
     * Gets the {@link DBConfig} for the clean source.
     *
     * @return {@link DBConfig}.
     */
    public DBConfig getSourceDBConfig() {
        return source;
    }

    /**
     * Gets the rule.
     *
     * @return rule.
     */
    public Rule getRule() {
        return rule;
    }
    // </editor-fold>

    private static List<String> toUppercase(List<String> values) {
        List<String> tmp = Lists.newArrayList();
        for (String val : values)
            tmp.add(val == null ? null : val.toUpperCase());
        return tmp;
    }

    @SuppressWarnings("unchecked")
    private static List<CleanPlan> create(
        CleanPlanJsonAdapter adapter,
        DBConfig nadeefDbConfig
    ) throws Exception {
        // a set which prevents generating new tables whenever encounters among
        // multiple rules.
        List<CleanPlan> result = Lists.newArrayList();
        List<Schema> schemas = Lists.newArrayList();
        SQLDialectBase dialectManager;

        // ----------------------------------------
        // parsing the source config
        // ----------------------------------------
        DBConfig dbConfig;
        SQLDialect sqlDialect;
        if (adapter.dbConfig.isCSV()) {
            dbConfig = NadeefConfiguration.getDbConfig();
            sqlDialect = dbConfig.getDialect();
        } else {
            sqlDialect = SQLDialectTools.getSQLDialect(adapter.dbConfig.type);
            dbConfig =
                new DBConfig.Builder()
                    .username(adapter.dbConfig.username)
                    .password(adapter.dbConfig.password)
                    .url(adapter.dbConfig.url)
                    .dialect(sqlDialect)
                    .build();
        }

        dialectManager = SQLDialectFactory.getDialectManagerInstance(sqlDialect);

        // ----------------------------------------
        // parsing the rules
        // ----------------------------------------
        ArrayList<Rule> rules = Lists.newArrayList();
        List<String> targetTableNames;
        List<String> fileNames = Lists.newArrayList();
        HashMap<String, String> copiedTables = Maps.newHashMap();

        for (RuleJsonAdapter ruleJson : adapter.rules) {
            schemas.clear();
            fileNames.clear();
            if (adapter.dbConfig.isCSV()) {
                // working with CSV
                List<String> fullFileNames = adapter.dbConfig.file;
                for (String fullFileName : adapter.dbConfig.file)
                    fileNames.add(Files.getNameWithoutExtension(fullFileName));

                if (ruleJson.hasTable())
                    targetTableNames = ruleJson.table;
                else
                    // if the target table names does not exist, we use
                    // default naming and only the first two tables are touched.
                    targetTableNames = fileNames;

                // convert all table names to Upper case for cross db compatibility.
                targetTableNames = toUppercase(targetTableNames);

                // source is a CSV file, dump it to NADEEF database.
                // This hashset is to prevent that tables are dumped for multiple times.
                for (int j = 0; j < targetTableNames.size(); j++) {
                    File file = CommonTools.getFile(fullFileNames.get(j));
                    String targetTable = targetTableNames.get(j);

                    // target table name already exists in the map.
                    if (!copiedTables.containsKey(targetTable)) {
                        String tableName =
                            CSVTools.dump(
                                nadeefDbConfig,
                                dialectManager,
                                file,
                                targetTable,
                                NadeefConfiguration.getAlwaysOverrideTable()
                            );
                        copiedTables.put(targetTable, tableName);
                    }
                    targetTable = copiedTables.get(targetTable);
                    targetTableNames.set(j, targetTable);
                    schemas.add(DBMetaDataTool.getSchema(dbConfig, targetTable));
                }
            } else {
                // working with database
                List<String> sourceTableNames = ruleJson.table;
                // convert all table names to Upper case for cross db compatibility
                sourceTableNames = toUppercase(sourceTableNames);

                for (String tableName : sourceTableNames)
                    if (!DBMetaDataTool.isTableExist(dbConfig, tableName))
                        throw new IllegalArgumentException(
                            "The specified table " +
                                tableName +
                                " cannot be found in the source database.");

                if (ruleJson.hasTarget())
                    targetTableNames = ruleJson.target;
                else {
                    // when user doesn't provide target tables we create a
                    // copy for them with default table names.
                    targetTableNames = Lists.newArrayList();
                    for (String sourceTableName : sourceTableNames) {
                        targetTableNames.add(sourceTableName + "_COPY");
                    }
                }

                // convert all table names to Upper case for cross db compatibility
                targetTableNames = toUppercase(targetTableNames);
                for (int j = 0; j < sourceTableNames.size(); j++) {
                    String sourceTable = sourceTableNames.get(j);

                    // when target table is as the same as the original table, skip copy
                    if (
                        !copiedTables.containsKey(sourceTable) &&
                        !sourceTableNames.get(j).equalsIgnoreCase(targetTableNames.get(j))
                    ) {
                        DBMetaDataTool.copy(
                            dbConfig,
                            dialectManager,
                            sourceTableNames.get(j),
                            targetTableNames.get(j)
                        );
                        copiedTables.put(sourceTable, sourceTable);
                    }

                    schemas.add(DBMetaDataTool.getSchema(dbConfig, targetTableNames.get(j)));
                }
            }

            Rule rule;
            String ruleName = ruleJson.name;
            if (!ruleJson.hasName())
                // generate default rule name when it is not provided by the user, and
                // distinguished by the value of the rule.
                ruleName =
                    "Rule" +
                        CommonTools.toHashCode(ruleJson.value.get(0) + targetTableNames.get(0));

            switch (ruleJson.type) {
                case "udf":
                    Class udfClass =
                        CommonTools.loadClass(ruleJson.value.get(0));
                    if (!Rule.class.isAssignableFrom(udfClass))
                        throw new IllegalArgumentException(
                            "The specified class is not a Rule class."
                        );

                    rule = (Rule) udfClass.newInstance();
                    // call internal initialization on the rule.
                    rule.initialize(ruleName, targetTableNames);
                    rules.add(rule);
                    break;
                default:
                    RuleBuilder ruleBuilder = NadeefConfiguration.tryGetRuleBuilder(ruleJson.type);
                    if (ruleBuilder != null)
                        rules.addAll(
                            ruleBuilder.name(ruleName)
                                .schema(schemas)
                                .table(targetTableNames)
                                .value(ruleJson.value)
                                .build()
                        );
                    else
                        tracer.error("Unknown Rule type: " + ruleJson.type, null);
                    break;
            }
        }

        for (Rule rule : rules)
            result.add(new CleanPlan(dbConfig, rule));
        return result;
    }
}
