#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_os_ostream.h"
#include <memory>
#include <utility>
#include <sstream>
#include <fstream>
#include <iostream>
#include <regex>


namespace headercvt{

std::stringstream
  types,
  func_decl,
  preprocessor_defines,
  not_handled;
llvm::raw_os_ostream
  types_ostream(types),
  func_decl_ostream(func_decl),
  preprocessor_defines_ostream(preprocessor_defines),
  not_handled_ostream(not_handled);
unsigned types_indentation = 0, func_decl_indentation = 0;

static constexpr char const* indent_str = "    ";

struct ostreams{
  std::vector<llvm::raw_ostream*> oss;
  ostreams(llvm::raw_ostream& os):oss{&os}{}
  template<typename T>
    llvm::raw_ostream& operator<<(T&& rhs){return (*oss.back()) << rhs;}
  operator llvm::raw_ostream&(){return *oss.back();}
  void push(llvm::raw_ostream& os){oss.emplace_back(&os);}
  void pop(){oss.pop_back();}
  struct auto_popper{
    ostreams* oss;
    auto_popper(ostreams& oss, llvm::raw_ostream& os):oss{&oss}{oss.push(os);}
    auto_popper(auto_popper&& other):oss{other.oss}{other.oss = nullptr;}
    ~auto_popper(){if(oss)oss->pop();}
  };
  auto_popper scoped_push(llvm::raw_ostream& os){return {*this, os};}
};

class preprocessor_defines_extractor : public clang::PPCallbacks{
private:
  unsigned Indentation;
  llvm::raw_ostream& Out;

  llvm::raw_ostream& Indent() { return Indent(Indentation); }
  llvm::raw_ostream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

public:
  preprocessor_defines_extractor(
      llvm::raw_ostream& Out,
      unsigned Indentation = 0
      ): Indentation(Indentation), Out(Out){

    Indent() << "cdef extern from \"CL/cl.h\"\n";
    this->Indentation++;
    Indent() << "cdef enum:\n";
    this->Indentation++;
  }

  void MacroDefined(const clang::Token& MacroNameTok, const clang::MacroDirective *MD) override{
    const clang::MacroDirective::Kind kind = MD->getKind();
    if (!(kind == clang::MacroDirective::Kind::MD_Define))
      return;

    const auto identifier = MacroNameTok.getIdentifierInfo()->getName().str();
    std::regex cl_macro_detector(R"(CL_.*)"); 
    if (!std::regex_match(identifier, cl_macro_detector))
      return;
    Indent() << identifier << "\n";
  }
};



  static clang::QualType getDeclType(clang::Decl* D) {
    if (clang::TypedefNameDecl* TDD = clang::dyn_cast<clang::TypedefNameDecl>(D))
      return TDD->getUnderlyingType();
    if (clang::ValueDecl* VD = clang::dyn_cast<clang::ValueDecl>(D))
      return VD->getType();
    return clang::QualType();
  }

  static clang::QualType GetBaseType(clang::QualType T) {
    // FIXME: This should be on the Type class!
    clang::QualType BaseType = T;
    while (!BaseType->isSpecifierType()) {
      if (const clang::PointerType *PTy = BaseType->getAs<clang::PointerType>())
        BaseType = PTy->getPointeeType();
      else if (const clang::BlockPointerType *BPy = BaseType->getAs<clang::BlockPointerType>())
        BaseType = BPy->getPointeeType();
      else if (const clang::ArrayType* ATy = clang::dyn_cast<clang::ArrayType>(BaseType))
        BaseType = ATy->getElementType();
      else if (const clang::FunctionType* FTy = BaseType->getAs<clang::FunctionType>())
        BaseType = FTy->getReturnType();
      else if (const clang::VectorType *VTy = BaseType->getAs<clang::VectorType>())
        BaseType = VTy->getElementType();
      else if (const clang::ReferenceType *RTy = BaseType->getAs<clang::ReferenceType>())
        BaseType = RTy->getPointeeType();
      else if (const clang::AutoType *ATy = BaseType->getAs<clang::AutoType>())
        BaseType = ATy->getDeducedType();
      else if (const clang::ParenType *PTy = BaseType->getAs<clang::ParenType>())
        BaseType = PTy->desugar();
      else
        // This must be a syntax error.
        break;
    }
    return BaseType;
  }



class simple_vardecl_printer : public clang::DeclVisitor<simple_vardecl_printer>{
  llvm::raw_ostream &Out;
  clang::PrintingPolicy Policy;
  // const clang::ASTContext &Context;
  unsigned Indentation;
public:
  simple_vardecl_printer(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
      const clang::ASTContext &, unsigned Indentation = 0)
    : Out(Out), Policy(Policy), Indentation(Indentation)
  {}

  void VisitParmVarDecl(clang::ParmVarDecl *D) {
    VisitVarDecl(D);
  }
  void VisitVarDecl(clang::VarDecl *D) {
    clang::QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    printDeclType(T, D->getName());
  }
  void printDeclType(clang::QualType T, llvm::StringRef DeclName, bool=false) {
    T.print(Out, Policy, DeclName, Indentation);
  }
};


class typedef_printer : public clang::DeclVisitor<typedef_printer>{
  llvm::raw_ostream &Out;
  clang::PrintingPolicy Policy;
  unsigned Indentation;

public:
  typedef_printer(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
              const clang::ASTContext &, unsigned Indentation = 0)
      : Out(Out), Policy(Policy), Indentation(Indentation) {}

  llvm::raw_ostream& Indent() { return Indent(Indentation); }
  llvm::raw_ostream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

  void VisitTypedefDecl(clang::TypedefDecl* D) {
    Indent();
    if (!Policy.SuppressSpecifiers) {
      Out << "ctypedef ";

      if (D->isModulePrivate())
        Out << "__module_private__ ";
    }

    clang::PrintingPolicy SubPolicy(Policy);
    SubPolicy.PolishForDeclaration = 1;
    auto Ty = D->getTypeSourceInfo()->getType().getUnqualifiedType();
    auto Typtr = Ty.getTypePtr();

    if (auto attrtyptr = clang::dyn_cast<clang::VectorType>(Typtr)){
      Out << attrtyptr->getElementType().getAsString();
    } else{
      Out << Ty.getAsString();
    }

    Out << " " << D->getName();
    Out << "\n";
  }

};




class grouped_typedef_struct_printer : public clang::DeclVisitor<grouped_typedef_struct_printer>{
	llvm::raw_ostream &Out;
	clang::PrintingPolicy Policy;
	unsigned Indentation;

	public:
	grouped_typedef_struct_printer(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
			const clang::ASTContext &, unsigned Indentation = 0)
		: Out(Out), Policy(Policy), Indentation(Indentation) {
		}

  void visit_group_struct_decl(clang::RecordDecl *RD, clang::TypedefDecl *TDD){
    if(!RD->isCompleteDefinition()){
      Indent()
        << "cdef struct "
        << RD->getName()
        << ":\n"
        ;
      Indentation++;
      Indent() << "pass\n";
      Indentation--;
      Indent() 
        << "ctypedef "
        << TDD->getTypeSourceInfo()->getType().getUnqualifiedType().getAsString()
        << " "
        << TDD->getName()
        << "\n";
    }
    else{
      Indent()
        << "ctypedef struct "
        << TDD->getName()
        << ":\n"
        ;
      Indentation++;
      if( RD->field_empty() ){
        Indent() << "pass\n";
      }else{
        for(auto itr = RD->field_begin(); itr != RD->field_end(); ++itr){
          VisitFieldDecl(*itr);
          Out << "\n";
        }
      }
    }
  }
	void VisitFieldDecl(clang::FieldDecl *D) {
		Indent() << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
			stream(Policy, D->getName(), Indentation);
	}

	llvm::raw_ostream& Indent() { return Indent(Indentation); }
	llvm::raw_ostream& Indent(unsigned Indentation) {
		for (unsigned i = 0; i != Indentation; ++i)
			Out << indent_str;
		return Out;
	}

};

class funcdecl_printer : public clang::DeclVisitor<funcdecl_printer>{
  llvm::raw_ostream &Out;
  clang::PrintingPolicy Policy;
  const clang::ASTContext &Context;
  unsigned Indentation;

public:
  funcdecl_printer(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
              const clang::ASTContext &Context, unsigned Indentation = 0)
      : Out(Out), Policy(Policy), Context(Context), Indentation(Indentation) {}


  llvm::raw_ostream& Indent() { return Indent(Indentation); }
  llvm::raw_ostream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }


  void prettyPrintAttributes(clang::Decl) {
    return;
  }


  void VisitFunctionDecl(clang::FunctionDecl *D) {
    auto const function_name = D->getNameInfo().getAsString();
    std::regex cl_function_detector(R"(cl[A-Z].*)");
    if (!std::regex_match(function_name, cl_function_detector))
      return;


    clang::CXXConstructorDecl *CDecl = clang::dyn_cast<clang::CXXConstructorDecl>(D);
    clang::CXXConversionDecl *ConversionDecl = clang::dyn_cast<clang::CXXConversionDecl>(D);
    clang::CXXDeductionGuideDecl *GuideDecl = clang::dyn_cast<clang::CXXDeductionGuideDecl>(D);

    Policy.SuppressSpecifiers = true;
    if (!Policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
        case clang::SC_None: break;
        case clang::SC_Extern: Out << "extern "; break;
        case clang::SC_Static: Out << "static "; break;
        case clang::SC_PrivateExtern: Out << "__private_extern__ "; break;
        case clang::SC_Auto: case clang::SC_Register:
                               llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  Out << "inline ";
      if (D->isVirtualAsWritten()) Out << "virtual ";
      if (D->isModulePrivate())    Out << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted()) Out << "constexpr ";
      if ((CDecl && CDecl->isExplicitSpecified()) ||
          (ConversionDecl && ConversionDecl->isExplicitSpecified()) ||
          (GuideDecl && GuideDecl->isExplicitSpecified()))
        Out << "explicit ";
    }
    Policy.SuppressSpecifiers = false;

    clang::PrintingPolicy SubPolicy(Policy);
    SubPolicy.SuppressSpecifiers = false;
    std::string Proto;
    Indent();

    if (Policy.FullyQualifiedName) {
      Proto += D->getQualifiedNameAsString();
    } else {
      if (!Policy.SuppressScope) {
        if (const clang::NestedNameSpecifier *NS = D->getQualifier()) {
          llvm::raw_string_ostream OS(Proto);
          NS->print(OS, Policy);
        }
      }
      Proto += D->getNameInfo().getAsString();
    }

    clang::QualType Ty = D->getType();
    while (const clang::ParenType *PT = clang::dyn_cast<clang::ParenType>(Ty)) {
      Proto = '(' + Proto + ')';
      Ty = PT->getInnerType();
    }

    if (const clang::FunctionType *AFT = Ty->getAs<clang::FunctionType>()) {
      const clang::FunctionProtoType *FT = nullptr;
      if (D->hasWrittenPrototype())
        FT = clang::dyn_cast<clang::FunctionProtoType>(AFT);

      Proto += "(";
      if (FT) {
        llvm::raw_string_ostream POut(Proto);
        simple_vardecl_printer ParamPrinter(POut, SubPolicy, Context, Indentation);

        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) POut << ", ";
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        }

        if (FT->isVariadic()) {
          if (D->getNumParams()) POut << ", ";
          POut << "...";
        }
      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }

      Proto += ")";

      if (FT) {
        if (FT->isConst())
          Proto += " const";
        if (FT->isVolatile())
          Proto += " volatile";
        if (FT->isRestrict())
          Proto += " restrict";

        switch (FT->getRefQualifier()) {
          case clang::RQ_None:
            break;
          case clang::RQ_LValue:
            Proto += " &";
            break;
          case clang::RQ_RValue:
            Proto += " &&";
            break;
        }
      }

      if (CDecl) {
        if (!Policy.TerseOutput)
          ;
          // PrintConstructorInitializers(CDecl, Proto);
      } else if (!ConversionDecl && !clang::isa<clang::CXXDestructorDecl>(D)) {
        if (FT && FT->hasTrailingReturn()) {
          if (!GuideDecl)
            Out << "auto ";
          Out << Proto << " -> ";
          Proto.clear();
        }
        AFT->getReturnType().print(Out, Policy, Proto);
        Proto.clear();
      }
      Out << Proto;
    } else {
      Ty.print(Out, Policy, Proto);
    }

    if (D->isPure())
      Out << " = 0";
    else if (D->isDeletedAsWritten())
      Out << " = delete";
    else if (D->isExplicitlyDefaulted())
      Out << " = default";
    else if (D->doesThisDeclarationHaveABody()) {
      if (!Policy.TerseOutput) {
        if (!D->hasPrototype() && D->getNumParams()) {
          // This is a K&R function definition, so we need to print the
          // parameters.
          Out << '\n';
          simple_vardecl_printer ParamPrinter(Out, SubPolicy, Context, Indentation);
          Indentation += Policy.Indentation;
          for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
            Indent();
            ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
            Out << ";\n";
          }
          Indentation -= Policy.Indentation;
        } else
          Out << ' ';

        if (D->getBody())
          D->getBody()->printPretty(Out, nullptr, SubPolicy, Indentation);
      } else {
        if (!Policy.TerseOutput && clang::isa<clang::CXXConstructorDecl>(*D))
          Out << " {}";
      }
    }

    Out << "\n";
  }

};


class general_decl_visitor : public clang::DeclVisitor<general_decl_visitor>{
  llvm::raw_ostream &Out;
  clang::PrintingPolicy Policy;
  const clang::ASTContext &Context;
  unsigned Indentation;

  public:
  general_decl_visitor(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
      const clang::ASTContext &Context, unsigned Indentation = 0)
    : Out(Out), Policy(Policy), Context(Context), Indentation(Indentation) {

    }

  llvm::raw_ostream& Indent() { return Indent(Indentation); }
  llvm::raw_ostream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

  void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D) {
    VisitDeclContext(D, false);
  }
  void VisitDeclContext(clang::DeclContext *DC, bool Indent) {
    if (Policy.TerseOutput)
      return;

    if (Indent)
      Indentation += Policy.Indentation;

    clang::SmallVector<clang::Decl*, 2> Decls;
    for (clang::DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
        D != DEnd; ++D) {

      // Skip over implicit declarations in pretty-printing mode.
      if (D->isImplicit())
        continue;

      // Don't print implicit specializations, as they are printed when visiting
      // corresponding templates.
      if (auto FD = clang::dyn_cast<clang::FunctionDecl>(*D))
        if (FD->getTemplateSpecializationKind() == clang::TSK_ImplicitInstantiation &&
            !clang::isa<clang::ClassTemplateSpecializationDecl>(DC))
          continue;

      // The next bits of code handle stuff like "struct {int x;} a,b"; we're
      // forced to merge the declarations because there's no other way to
      // refer to the struct in question.  When that struct is named instead, we
      // also need to merge to avoid splitting off a stand-alone struct
      // declaration that produces the warning ext_no_declarators in some
      // contexts.
      //
      // This limited merging is safe without a bunch of other checks because it
      // only merges declarations directly referring to the tag, not typedefs.
      //
      // Check whether the current declaration should be grouped with a previous
      // non-free-standing tag declaration.
      clang::QualType CurDeclType = getDeclType(*D);
      if (!Decls.empty() && !CurDeclType.isNull()) {
        clang::QualType BaseType = GetBaseType(CurDeclType);
        if (!BaseType.isNull() && clang::isa<clang::ElaboratedType>(BaseType)){
          BaseType = clang::cast<clang::ElaboratedType>(BaseType)->getNamedType();
        }
        if (!BaseType.isNull() && clang::isa<clang::TagType>(BaseType) &&
            clang::cast<clang::TagType>(BaseType)->getDecl() == Decls[0]) {
          Decls.push_back(*D);
          continue;
        }
      }

      // If we have a merged group waiting to be handled, handle it now.
      if (!Decls.empty())
        ProcessDeclGroup(Decls);

      // If the current declaration is not a free standing declaration, save it
      // so we can merge it with the subsequent declaration(s) using it.
      if (clang::isa<clang::TagDecl>(*D) && !clang::cast<clang::TagDecl>(*D)->isFreeStanding()) {
        Decls.push_back(*D);
        continue;
      }


      this->Indent();
      if (clang::isa<clang::FunctionDecl>(*D)){
        funcdecl_printer FuncDeclPrinter(func_decl_ostream, Policy, Context, func_decl_indentation);
        FuncDeclPrinter.Visit(clang::cast<clang::FunctionDecl>(*D));
        func_decl_ostream.flush();
      }

      if (clang::isa<clang::TypedefDecl>(*D)){
        typedef_printer TypedefPrinter(types_ostream, Policy, Context, types_indentation);
        TypedefPrinter.Visit(*D);
        types_ostream.flush();
      }

    } // end of in-declcontext iteration

    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    if (Indent)
      Indentation -= Policy.Indentation;
  }

  void ProcessDeclGroup(clang::SmallVectorImpl<clang::Decl*>& Decls) {
    this->Indent();
    llvm::raw_os_ostream declgroup_ostream(types);
    printTypedefStructDeclGroup(Decls.data(), Decls.size(), declgroup_ostream);
    Decls.clear();
  }

  void printTypedefStructDeclGroup(clang::Decl** Begin, unsigned NumDecls, llvm::raw_ostream &Out) {
    if(NumDecls == 1) return;

    clang::TagDecl* TD = clang::dyn_cast<clang::TagDecl>(*Begin);
    clang::TypedefDecl* TDD = clang::dyn_cast<clang::TypedefDecl>(*(Begin+1));
    if (TD && TDD){
      clang::TagTypeKind const kind = TD -> getTagKind();
      switch (kind){
        case clang::TagTypeKind::TTK_Struct:
          // Group は `typedef struct {field_name, ...} type_name;` 宣言グループ
/*
|-RecordDecl struct definition      <- Begin, TD
| `-FieldDecl field_name 'int [2]'
|-TypedefDecl referenced type_name 'struct type_name':'type_name'  <- TDD
| `-ElaboratedType 'struct type_name' sugar
|   `-RecordType 'type_name'
|     `-Record ''
*/
          // または `typedef _hoge* hoge_t;` (_tag宣言なし) 宣言グループ
/*
|-RecordDecl struct _hoge               <- Begin, TD
|-TypedefDecl hoge_t 'struct _hoge *'   <- TDD
| `-PointerType 0x5536680 'struct _hoge *'
|   `-ElaboratedType 0x5536620 'struct _hoge' sugar
|     `-RecordType 0x5536600 'struct _hoge'
|       `-Record 0x54dd690 '_hoge'
*/
          {
            auto tdp = grouped_typedef_struct_printer(Out, Policy, Context, types_indentation);
            clang::RecordDecl* RD = clang::dyn_cast<clang::RecordDecl>(TD);
            tdp.visit_group_struct_decl( RD, TDD );
            return;
          }
        case clang::TagTypeKind::TTK_Union :
          Out << "# union declaration ignored\n";
          return;
        case clang::TagTypeKind::TTK_Interface :
        case clang::TagTypeKind::TTK_Class :
        case clang::TagTypeKind::TTK_Enum :
					return;
      }
    }
  }
};

namespace registrar{

class ast_consumer : public clang::ASTConsumer{
  std::unique_ptr<general_decl_visitor> visit;
 public:
  explicit ast_consumer(clang::CompilerInstance& ci) : visit{
    new general_decl_visitor{
      not_handled_ostream,
        ci.getASTContext().getPrintingPolicy(),
        ci.getASTContext()
    }
  } // initializer
  { // body
    ci.getPreprocessor().addPPCallbacks(llvm::make_unique<preprocessor_defines_extractor>(
            preprocessor_defines_ostream
          ));
  }
  virtual void HandleTranslationUnit(clang::ASTContext& context)override{
    visit->Visit(context.getTranslationUnitDecl());
  }
};

struct ast_frontend_action : clang::SyntaxOnlyAction{
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& ci, clang::StringRef)override{
    return llvm::make_unique<ast_consumer>(ci);
  }
};

}

}

int main(int argc, const char** argv){
  llvm::cl::OptionCategory tool_category("headercvt options");
  llvm::cl::extrahelp common_help(clang::tooling::CommonOptionsParser::HelpMessage);
  std::vector<const char*> params;
  params.reserve(argc+1);
  std::copy(argv, argv+argc, std::back_inserter(params));
  params.emplace_back("-xc");
  params.emplace_back("-w");
  params.emplace_back("-Wno-narrowing");
  clang::tooling::CommonOptionsParser options_parser(argc = static_cast<int>(params.size()), params.data(), tool_category);
  clang::tooling::ClangTool tool(options_parser.getCompilations(), options_parser.getSourcePathList());

  {
    headercvt::func_decl_ostream << "cdef extern from \"CL/cl.h\"\n";
    headercvt::func_decl_indentation ++;
  }
  {
    headercvt::types_ostream << "cdef extern from \"CL/cl.h\"\n";
    headercvt::types_indentation ++;
  }

  auto const result_value = tool.run(clang::tooling::newFrontendActionFactory<headercvt::registrar::ast_frontend_action>().get());

  headercvt::func_decl_ostream.flush();
  headercvt::types_ostream.flush();
  headercvt::not_handled_ostream.flush();
  headercvt::preprocessor_defines_ostream.flush();
  std::cout << "\n\n func_decl ---------------------------------------------------\n";
  std::cout << headercvt::func_decl.str() << std::endl;
  std::cout << "\n\n preprocessor_defines ---------------------------------------------------\n";
  std::cout << headercvt::preprocessor_defines.str() << std::endl;
  std::cout << "\n\n types ---------------------------------------------------\n";
  std::cout << headercvt::types.str() << std::endl;
  std::cout << "\n\n not handled ---------------------------------------------------\n";
  std::cout << headercvt::not_handled.str() << std::endl;

  return result_value;
}
