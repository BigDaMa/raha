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

define([
    "mvc/editor/NodeEditor",
    "requester",
    "text!mvc/editor/template/er.template.html",
    "text!mvc/editor/template/property.template.html"
], function (NodeEditor, Requester, ERTemplate, TableTemplate) {
    "use strict";
    var int2op = {
        0: 'EQ',
        1: 'ED',
        2: 'LS',
        3: 'QG',
        4: 'SD'
    };

    var op2int = {
        "EQ" : 0,
        "ED" : 1,
        "LS" : 2,
        "QG" : 3,
        "SD" : 4
    };

    var int2cmp = {
        0: '=',
        1: '>',
        2: '<',
        3: '>=',
        4: '<=',
        5: '!='
    };

    var cmp2int = {
        '=' : 0,
        '>' : 1,
        '<' : 2,
        '>=': 3,
        '<=': 4,
        '!=': 5
    };

    function parse(str) {
        var reg =
            new RegExp(
            "(EQ|ED|LS|QG|SD)\\((\\w+)\\.(\\w+),(\\w+)\\.(\\w+)\\)(=|>|<|>=|<=|!=)(\\d+\\.?\\d*)");
        if (reg.test(str)) {
            var matches = reg.exec(str);
            return {
                table1 : matches[2],
                column1 : matches[3],
                table2 : matches[4],
                column2 : matches[5],
                property: {
                    op: op2int[matches[1]],
                    cmp: cmp2int[matches[6]],
                    val: matches[7]
                }
            };
        }
        return null;
    }

    function EREditor(dom, table1, table2, rule) {
        this.dom = dom;
        this.tableName1 = table1;
        this.tableName2 = _.isEmpty(table2) ? table1 : table2;
        this.rule = rule;
        this.editor = null;
        this.predicates = [];
    }

    EREditor.prototype.render = function () {
        var promise1 = null;
        var __ = this;
        if (_.isEmpty(this.tableName1)) {
            promise1 = null;
        } else {
            promise1 = Requester.getTableSchema(this.tableName1, {
                success: function (data) {
                    __.table1 = { name: __.tableName1, columns: data.schema};
                }
            });
        }

        var promise2 = null;
        if (_.isEmpty(this.tableName2)) {
            promise2 = null;
            this.tableName2 = null;
        } else {
            promise2 = Requester.getTableSchema(this.tableName2, {
                success: function (data) {
                    __.table2 = { name: __.tableName2, columns: data.schema};
                }
            });
        }

        $.when(promise1, promise2).then(function () {
            var conns = [];
            if (__.rule.code) {
                var ps = __.rule.code.split("\n");
                for (var i = 0; i < ps.length; i ++) {
                    var tmp = parse(ps[i]);
                    if (tmp) {
                        var pin0, pin1;
                        if (tmp.table1 === __.table1.name) {
                            pin0 = new NodeEditor.CreatePin(
                                __.table1.name,
                                tmp.column1,
                                _.indexOf(__.table1.columns, tmp.column1)
                            );

                            pin1 = new NodeEditor.CreatePin(
                                __.table2.name,
                                tmp.column2,
                                _.indexOf(__.table2.columns, tmp.column2)
                            );
                        } else {
                            pin0 = new NodeEditor.CreatePin(
                                __.table2.name,
                                tmp.column2,
                                _.indexOf(__.table2.columns, tmp.column2)
                            );

                            pin1 = new NodeEditor.CreatePin(
                                __.table1.name,
                                tmp.column1,
                                _.indexOf(__.table1.columns, tmp.column1)
                            );
                        }

                        conns.push(new NodeEditor.CreateConnection(pin0, pin1, tmp.property));
                    }
                }
            }

            $(__.dom).html(_.template(ERTemplate) ({}));
            __.nodeEditor = new NodeEditor.Create(
                $(__.dom).find('#playground'),
                __.table1,
                __.table2,
                conns,
                function () { return { op: 0, val: 1.0, cmp: 0 }; }
            );

            // wrap up the right closure
            __.nodeEditor.onLineAdded(function (d) { __.drawProperty(d); });
            __.nodeEditor.render();
            __.drawProperty();
        });
    };

    EREditor.prototype.drawProperty = function () {
        this.predicates = this.nodeEditor.getConnection();

        var __ = this;
        var trs =
            d3.select("#ruleTableBody")
              .selectAll("tr")
              .data(this.predicates, function (d) { return d.id; });

        trs.exit().remove();
        trs.enter().append("tr");
        trs.html(function (d, i) {
            return _.template(
                TableTemplate, {
                    id: i,
                    left: d.left.getName(),
                    right: d.right.getName(),
                    op: d.property.op,
                    cmp: d.property.cmp,
                    value: d.property.val
                });
        }).each(function (d, i) {
            $(__.dom).find("#del" + i).on("click", function () {
                console.log('del' + i);
                __.predicates.splice(i, 1);
                __.nodeEditor.updateConnection(__.predicates);
                __.drawProperty();
            });

            $(__.dom).find("#op" + i).change(function () {
                console.log('op' + i);
                __.predicates[i].property.op = $(__.dom).find("#op" + i).val();
                __.nodeEditor.updateConnection(__.predicates);
                __.drawProperty();
            });

            $(__.dom).find("#cmp" + i).change(function () {
                console.log('cmp' + i);
                __.predicates[i].property.cmp = $(__.dom).find("#cmp" + i).val();
                __.nodeEditor.updateConnection(__.predicates);
            });

            $(__.dom).find("#v" + i).change(function () {
                console.log('v' + i);
                __.predicates[i].property.val = $(__.dom).find("#v" + i).val();
                __.nodeEditor.updateConnection(__.predicates);
            });
        });
    };

    EREditor.prototype.val = function () {
        if (_.isEmpty(this.predicates)) {
            return null;
        }

        var cmd = "";
        _.each(this.predicates, function (d) {
            cmd += int2op[d.property.op];
            cmd += "(";
            cmd += d.left.getName();
            cmd += ",";
            cmd += d.right.getName();
            cmd += ")";
            cmd += int2cmp[d.property.cmp];
            cmd += d.property.val;
            cmd += "\n";
        });

        return cmd;
    };

    return {
        Create: EREditor
    };
});