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

package qa.qcri.nadeef.ruleext;

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Operation;
import qa.qcri.nadeef.core.datamodel.Predicate;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.tools.CommonTools;

import java.io.File;
import java.util.Collection;
import java.util.List;

/**
 * Template engine for CFD rule.
 */
public class CFDRuleBuilder extends RuleBuilder {
    protected List<String>                 lhs;
    protected List<String>                 rhs;
    protected List<List<Predicate>> filterExpressions;
    protected STGroupFile                  singleSTGroup;
    protected STGroupFile                  pairSTGroup;

    @Override
    public Collection<File> generate() throws Exception {
        List<File> result = Lists.newArrayList();
        singleSTGroup = new STGroupFile(
            "qa/qcri/nadeef/ruleext/template/SingleCFDRuleBuilder.stg",
            '$', '$'
        );

        pairSTGroup = new STGroupFile(
            "qa/qcri/nadeef/ruleext/template/PairCFDRuleBuilder.stg",
            '$', '$'
        );

        ST sst = singleSTGroup.getInstanceOf("cfdTemplate");
        ST pst = pairSTGroup.getInstanceOf("cfdTemplate");
        ST targetST;

        sst.add("leftHandSide", lhs);
        pst.add("leftHandSide", lhs);

        for (int i = 0; i < filterExpressions.size(); i++) {
            List<Predicate> filters = filterExpressions.get(i);
            ST leftExpressionST = singleSTGroup.getInstanceOf("lExpressionItem");
            List<String> leftExpressions = Lists.newArrayList();

            // Pass over items of lhs
            for (int j = 0; j < lhs.size(); j++) {
                Predicate expression = filters.get(j);
                String eValue = (String)expression.getValue();
                if (!eValue.equals("_")) {
                    leftExpressionST.add("columnName", expression.getLeft()
                            .getFullColumnName());
                    leftExpressionST.add("value", eValue);
                    leftExpressions.add(leftExpressionST.render());
                    leftExpressionST.remove("columnName");
                    leftExpressionST.remove("value");
                }
            }

            // Normalize CFD; create a file for each column in RHS
            for (int j = 0; j < rhs.size(); j++) {
                String rhsCol = rhs.get(j);

                Predicate expression = filters.get(j + lhs.size());
                String eValue = (String)expression.getValue();
                if (!eValue.equals("_")) {
                    ST rightExpressionST = singleSTGroup.getInstanceOf("rExpressionItem");
                    rightExpressionST.add("columnName", expression.getLeft().getFullColumnName());
                    rightExpressionST.add("value", eValue);
                    List<String> rightExpressions = Lists.newArrayList();
                    rightExpressions.add(rightExpressionST.render());
                    rightExpressionST.remove("columnName");
                    rightExpressionST.remove("value");

                    sst.add("lExpression", leftExpressions);
                    sst.add("rExpression", rightExpressions);
                    targetST = sst;
                } else {
                    pst.add("lExpression", leftExpressions);
                    targetST = pst;
                }

                targetST.add("rightHandSide", rhsCol);

                if (Strings.isNullOrEmpty(originalRuleName)) {
                    ruleName = "DefaultCFD"
                            + CommonTools.toHashCode(value.get(i + 1)) + "_"
                            + rhsCol.substring(rhsCol.lastIndexOf('.') + 1)
                            + "_" + i;
                } else {
                    // remove all the empty spaces to make it a valid class
                    // name.
                    ruleName = originalRuleName.replace(" ", "") + "_"
                            + rhsCol.substring(rhsCol.lastIndexOf('.') + 1)
                            + "_" + i;
                }

                targetST.add("CFDName", ruleName);

                File outputFile = getOutputFile();
                targetST.write(outputFile, null);
                result.add(outputFile);

                targetST.remove("CFDName");
                targetST.remove("lExpression");
                if (targetST == sst) {
                    targetST.remove("rExpression");
                }
                targetST.remove("rightHandSide");
            }
        }
        return result;
    }

    /**
     * Generates and compiles the rule .class file without loading it.
     * 
     * @return Output class file.
     */
    @Override
    public Collection<File> compile() throws Exception {
        Collection<File> result = Lists.newArrayList();
        Collection<File> javaFiles = generate();
        for (File outputFile : javaFiles) {
            String fullPath = outputFile.getAbsolutePath();
            // skip compiling if the .class file already exists.
            File classFile = new File(fullPath.replace(".java", ".class"));
            if (classFile.exists() || CommonTools.compileFile(outputFile) != null) {
                result.add(classFile);
            }
        }

        return result;
    }

    /**
     * Interpret a rule from input text stream.
     */
    @Override
    protected void parse() {
        lhs = Lists.newArrayList();
        rhs = Lists.newArrayList();
        filterExpressions = Lists.newArrayList();

        String head = value.get(0);
        String[] splits = head.trim().split("\\|");
        String split;
        if (splits.length != 2) {
            throw new IllegalArgumentException("Invalid rule description "
                    + head);
        }

        // parse the LHS
        String[] lhsSplits = splits[0].split(",");
        // use the first one as default table name.
        String defaultTable = tableNames.get(0);
        for (int i = 0; i < lhsSplits.length; i++) {
            split = lhsSplits[i].trim().toLowerCase();
            if (Strings.isNullOrEmpty(split)) {
                throw new IllegalArgumentException("Invalid rule description "
                        + head);
            }

            if (!Strings.isNullOrEmpty(defaultTable)
                    && !CommonTools.isValidColumnName(split)) {
                lhs.add(new Column(defaultTable, split).getFullColumnName());
            } else {
                lhs.add(split);
            }
        }

        // parse the RHS
        String[] rhsSplits = splits[1].trim().split(",");
        for (int i = 0; i < rhsSplits.length; i++) {
            split = rhsSplits[i].trim().toLowerCase();
            if (split.isEmpty()) {
                throw new IllegalArgumentException("Invalid rule description "
                        + head);
            }

            if (!Strings.isNullOrEmpty(defaultTable)
                    && !CommonTools.isValidColumnName(split)) {
                rhs.add(new Column(defaultTable, split).getFullColumnName());
            } else {
                rhs.add(split);
            }
        }

        // parse condition line
        // TODO: currently we recreate a new FDRule per condition line, but
        // it would have more optimizations based on a buck of lines.
        for (int i = 1; i < value.size(); i++) {
            List<Predicate> filter = Lists.newArrayList();
            String line = value.get(i);
            if (Strings.isNullOrEmpty(line)) {
                continue;
            }

            splits = line.trim().split(",");
            if (splits.length != lhs.size() + rhs.size()) {
                throw new IllegalArgumentException("Invalid rule description "
                        + line);
            }

            String curColumn;
            for (int j = 0; j < splits.length; j++) {
                split = splits[j].trim().toLowerCase();
                if (j < lhs.size()) {
                    curColumn = lhs.get(j);
                } else {
                    curColumn = rhs.get(j - lhs.size());
                }

                if (Strings.isNullOrEmpty(split)) {
                    throw new IllegalArgumentException(
                            "Invalid rule description " + line);
                }

                Predicate newFilter =
                    new Predicate.PredicateBuilder()
                        .left(new Column(curColumn))
                        .isSingle()
                        .op(Operation.EQ)
                        .constant(split)
                        .build();
                filter.add(newFilter);
            }
            filterExpressions.add(filter);
        }
    }

    /**
     * Gets of LHS.
     * 
     * @return LHS.
     */
    public List<String> getLhs() {
        return lhs;
    }

    /**
     * Gets of RHS.
     * 
     * @return RHS.
     */
    public List<String> getRhs() {
        return rhs;
    }

    /**
     * Gets of exists filter expressions.
     * 
     * @return filter expressions.
     */
    public List<List<Predicate>> getFilterExpressions() {
        return filterExpressions;
    }
}
