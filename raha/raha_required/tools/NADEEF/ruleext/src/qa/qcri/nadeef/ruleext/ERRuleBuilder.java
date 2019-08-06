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
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.RuleBuilder;
import qa.qcri.nadeef.tools.CommonTools;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Template engine for ER rule.
 */
public class ERRuleBuilder extends RuleBuilder  {
    private static final Pattern pattern =
        Pattern.compile(
            "\\s*(EQ|ED|LS|QG|SD)\\s*" +
            "\\(\\s*([a-zA-Z_]\\w*)\\.([a-zA-Z_]\\w*)\\s*," +
            "\\s*([a-zA-Z_]\\w*)\\.([a-zA-Z_]\\w*)\\s*\\)" +
            "\\s*(>|<|<=|>=|=|!=)\\s*(\\d+\\.?\\d*)\\s*"
        );

    @Override
    public Collection<File> compile() throws IOException {
        File inputFile = generate().iterator().next();
        // check whether the class file is already there, if so we skip the compiling phrase.
        String fullPath = inputFile.getAbsolutePath();
        File classFile = new File(fullPath.replace(".java", ".class"));
        boolean alwaysCompile = NadeefConfiguration.getAlwaysCompile();
        if (alwaysCompile || !classFile.exists()) {
            CommonTools.compileFile(inputFile);
        }

        return Lists.newArrayList(classFile);
    }

    @Override
    public Collection<File> generate() throws IOException {
        List<String> predicates = Lists.newArrayList();
        for (String predicateText : value) {
            Matcher matcher = pattern.matcher(predicateText);
            if (!matcher.matches()) {
                throw new IllegalArgumentException(
                    "Given text " + predicateText + " is not a valid predicate text.");
            }

            String metric = matcher.group(1);
            String leftTable = matcher.group(2);
            String leftAttribute = matcher.group(3);
            String rightTable = matcher.group(4);
            String rightAttribute = matcher.group(5);
            String op = matcher.group(6);
            if (op.equals("="))
                op = "==";
            String threshold = matcher.group(7);

            StringBuilder sb = new StringBuilder();
            String left =
                String.format("getValue(tuplePair, \"%s\", \"%s\", 0)", leftTable, leftAttribute);
            String right =
                String.format("getValue(tuplePair, \"%s\", \"%s\", 1)", rightTable, rightAttribute);

            switch (metric) {
                case "EQ":
                    sb.append("Metrics.getEqual");
                    break;
                case "ED":
                    sb.append("Metrics.getEuclideanDistance");
                    break;
                case "LS":
                    sb.append("Metrics.getLevenshtein");
                    break;
                case "QG":
                    sb.append("Metrics.getQGramsDistance");
                    break;
                case "SD":
                    sb.append("Metrics.getSoundex");
                    break;
                default:
                    throw new IllegalArgumentException("Unknown metric similarity function");
            }

            sb.append("(")
                .append(left)
                .append(",")
                .append(right)
                .append(")")
                .append(op)
                .append(threshold);
            predicates.add(sb.toString());
        }

        STGroupFile stFile =
            new STGroupFile("qa/qcri/nadeef/ruleext/template/ERRuleBuilder.stg", '$', '$');
        ST st = stFile.getInstanceOf("erTemplate");
        st.add("predicates", predicates);
        if (Strings.isNullOrEmpty(ruleName)) {
            ruleName = "DefaultER" + CommonTools.toHashCode(value.get(0));
        } else {
            // remove all the empty spaces to make it a valid class name.
            ruleName = originalRuleName.replace(" ", "");
        }

        st.add("ERName", ruleName);
        File outputFile = getOutputFile();
        st.write(outputFile, null);
        return Lists.newArrayList(outputFile);
    }
}
