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

package qa.qcri.nadeef.test.core;

import com.google.common.collect.Lists;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.*;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.CSVTools;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

import java.io.File;
import java.nio.charset.Charset;
import java.sql.Types;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * MemoryTable test.
 */
@RunWith(Parameterized.class)
public class MemoryTableTest extends NadeefTestBase {
    public MemoryTableTest(String testConfig_) {
        super(testConfig_);
    }

    private List<Tuple> testTuples;

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);

            Schema schema =
                new Schema.Builder()
                    .table("test")
                    .column("C", Types.VARCHAR)
                    .column("A", Types.VARCHAR)
                    .column("B", Types.VARCHAR)
                    .build();
            File dumpFile = TestDataRepository.getDumpTestCSVFile();
            List<String[]> content = CSVTools.read(dumpFile, ",");
            testTuples = Lists.newArrayList();
            List<byte[]> values = Lists.newArrayList();
            for (int i = 0; i < content.size(); i ++) {
                String[] tokens = content.get(i);
                for (String token : tokens) {
                    values.add(token.getBytes(Charset.forName("UTF-8")));
                }

                testTuples.add(new Tuple(i + 1, schema, Lists.newArrayList(values)));
                values.clear();
            }
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail();
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Test
    public void testProjection() {
        MemoryTable table = MemoryTable.of(testTuples);
        table.project("C");
        Assert.assertEquals(12, table.size());
        Tuple tuple = table.get(0);
        Set cellSets = tuple.getCells();
        Assert.assertEquals(1, cellSets.size());
        Cell cell = (Cell)cellSets.iterator().next();
        Assert.assertEquals("test.C", cell.getColumn().getFullColumnName());
    }

    @Test
    public void testFilter() {
        MemoryTable table = MemoryTable.of(testTuples);
        table.filter(
            Predicate.createEq(new Column("test", "C"), "c1")
        ).project(new Column("test", "C"));
        Assert.assertEquals(7, table.size());
        Tuple tuple = table.get(0);
        Set cellSets = tuple.getCells();
        Assert.assertEquals(1, cellSets.size());
        Cell cell = (Cell)cellSets.iterator().next();
        Assert.assertEquals("test.C", cell.getColumn().getFullColumnName());
    }

    @Test
    public void testGroup() {
        MemoryTable table = MemoryTable.of(testTuples);
        Collection<Table> result = table.groupOn("C");
        Assert.assertEquals(3, result.size());
        for (Table t : result) {
            Tuple tuple = t.get(0);
            String value =(String) tuple.get("c");
            switch (value) {
                case "c1":
                    Assert.assertEquals(7, t.size());
                    break;
                case "c3":
                    Assert.assertEquals(1, t.size());
                    break;
                case "c2":
                    Assert.assertEquals(4, t.size());
                    break;
            }

        }
    }
}
