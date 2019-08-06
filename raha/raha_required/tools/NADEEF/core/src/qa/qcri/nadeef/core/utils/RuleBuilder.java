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

package qa.qcri.nadeef.core.utils;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.Logger;
import sun.rmi.runtime.Log;

import java.io.File;
import java.net.URL;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Abstract class RuleBuilder represents the extension point of writing new
 * abstract rules (e.g. FD).
 */
public abstract class RuleBuilder {
    // <editor-fold desc="Private fields">
    protected List<String> value;
    protected String       originalRuleName;
    protected String       ruleName;
    protected List<String> tableNames;
    protected List<Schema> schemas;
    protected File         outputPath;

    // </editor-fold>

    public RuleBuilder() {
    }

    // <editor-fold desc="Public methods">
    /**
     * Sets the rule value.
     * 
     * @param value rule value.
     */
    public RuleBuilder value(String value) {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(value));
        this.value = Lists.newArrayList(value);
        parse();
        return this;
    }

    /**
     * Sets the source of the Rule.
     * 
     * @param schemas input schemas.
     */
    public RuleBuilder schema(List<Schema> schemas) {
        this.schemas = Preconditions.checkNotNull(schemas);
        return this;
    }

    /**
     * Sets the source of the Rule.
     * 
     * @param schema input schemas.
     */
    public RuleBuilder schema(Schema schema) {
        Preconditions.checkNotNull(schema);
        schemas = Lists.newArrayList(schema);
        return this;
    }

    /**
     * Sets the rule value.
     * 
     * @param values rule value.
     */
    public RuleBuilder value(List<String> values) {
        this.value = Preconditions.checkNotNull(values);
        parse();
        return this;
    }

    /**
     * Sets the table names.
     * 
     * @param tableNames table name.
     */
    public RuleBuilder table(List<String> tableNames) {
        Preconditions.checkArgument(tableNames != null && tableNames.size() > 0);
        this.tableNames = tableNames;
        return this;
    }

    /**
     * Sets the table names.
     * 
     * @param table table name.
     */
    public RuleBuilder table(String table) {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(table));
        this.tableNames = Lists.newArrayList(table);
        return this;
    }

    /**
     * Sets the output path.
     * 
     * @param outputPath output path.
     */
    public RuleBuilder out(File outputPath) {
        Preconditions.checkArgument(outputPath != null && outputPath.isDirectory());
        this.outputPath = outputPath;
        return this;
    }

    /**
     * Sets the rule name.
     * 
     * @param ruleName rule name.
     */
    public RuleBuilder name(String ruleName) {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(ruleName));
        this.originalRuleName = ruleName;
        this.ruleName = ruleName;
        return this;
    }

    /**
     * Generates the <code>Rule</code> class code.
     * 
     * @return generated rule class.
     */
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public Collection<Rule> build() throws Exception {
        List<Rule> result = Lists.newArrayList();
        Collection<File> outputFiles = compile();
        Logger tracer = Logger.getLogger(RuleBuilder.class);
        for (File outputFile : outputFiles) {
            Stopwatch stopwatch = Stopwatch.createStarted();
            String className =
                Files.getNameWithoutExtension(outputFile.getName());

            URL url =
                new URL(
                    "file://" + outputFile.getParent() + File.separator
                );
            Class ruleClass = CommonTools.loadClass(className, url);
            Rule rule = (Rule) ruleClass.getConstructor().newInstance();

            rule.initialize(
                Files.getNameWithoutExtension(outputFile.getName()),
                tableNames
            );
            result.add(rule);
            tracer.fine(
                "Rule file : " + outputFile.getAbsolutePath() +
                " is loaded in " +
                stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
            );
            stopwatch.stop();
        }
        return result;
    }

    /**
     * Generates and compiles the rule to the .class file.
     * 
     * @return Output class file.
     */
    public abstract Collection<File> compile() throws Exception;

    /**
     * Generates the .java file.
     *
     * @return Output class file.
     */
    public abstract Collection<File> generate() throws Exception;

    // </editor-fold>

    // <editor-fold desc="Protected Fields">

    /**
     * Gets the compiled output file.
     * 
     * @return output file.
     */
    protected File getOutputFile() {
        String currentPath;
        if (outputPath == null) {
            currentPath = NadeefConfiguration.getOutputPath().toString();
        } else {
            currentPath = outputPath.getAbsolutePath();
        }
        return new File(currentPath + File.separator + ruleName + ".java");
    }

    /**
     * Parse the value into a rule.
     */
    protected void parse() {
    }

    // </editor-fold>
}
