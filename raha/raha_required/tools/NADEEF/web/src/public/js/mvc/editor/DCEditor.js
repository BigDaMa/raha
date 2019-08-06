define([
    "requester",
    "text!mvc/editor/template/dc.template.html"
], function (Requester, Template) {
    "use strict";
    function DCEditor(dom, table1, table2, rule) {
        this.dom = dom;
        this.tableName1 = table1;
        this.tableName2 = table2;
        this.rule = rule;
    }

    DCEditor.prototype.render = function () {
        var promise = null;
        var __ = this;
        if (_.isEmpty(this.tableName1)) {
            promise = null;
        } else {
            promise = Requester.getTableSchema(this.tableName1, {
                success: function (data) {
                    __.table1 = { name: __.tableName1, columns: data.schema};
                },
                failure: function () {
                    console.log("Schema fetching failed.");
                }
            });
        }

        $.when(promise).then(function () {
            $(__.dom).html(_.template(Template, {
                code : __.rule.code,
                columns : _.isUndefined(__.table1) ? [] : __.table1.columns
            }));
        });
    };

    DCEditor.prototype.val = function () {
        return $(this.dom).find("#value").val();
    };

    return {
        Create: DCEditor
    };
});